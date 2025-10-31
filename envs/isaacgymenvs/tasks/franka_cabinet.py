import numpy as np
import os
import torch
import json

from isaacgym import gymtorch
from isaacgym import gymapi

from envs.isaacgymenvs.utils.torch_jit_utils import to_torch, tensor_clamp, quat_apply, torch_rand_float, tf_vector, tf_combine, tf_inverse, tf_apply, normalize, batch_dot, get_axis_params
from envs.isaacgymenvs.tasks.base.vec_task import VecTask, project_point_on_segment, is_under_valid_vel, mix_clone, mix_norm, ema_filter
from tf_utils import tf_inverse as tf_inverse_np, tf_combine as tf_combine_np
from copy import deepcopy


class FrankaCabinet(VecTask):
    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):
        self.cfg = cfg

        # Custom Variable Init
        self.custom_variable_init()
        self.num_props = self.cfg["env"]["numProps"] if self.cfg.get("num_props_eval", None) is None else self.cfg["num_props_eval"]

        # Create dicts to pass to reward function
        self.reward_settings = {
            "r_dist_scale": self.cfg["env"]["distRewardScale"],
            "r_rot_scale": self.cfg["env"]["rotRewardScale"],
            "r_around_handle_scale": self.cfg["env"]["aroundHandleRewardScale"],
            "r_open_scale": self.cfg["env"]["openRewardScale"],
            "r_finger_dist_scale": self.cfg["env"]["fingerDistRewardScale"],
        }

        self.dr_settings["noise"].update({
            # pos in meters, quat/dof in radians
            "drawer_handle_pos": [0., 0.01]
        })
        
        self.dr_settings["spatial"].update({
            "cabinet_pos": [0., self.start_position_noise],
        })

        # dimensions
        # obs include: drawer_handle_pos (3) + drawer_init_pos (3) + eef_pos (3) + eef_rot(4) + gripper_mode (1) [+ q_arm (7)]
        self.cfg["env"]["numObservations"] = 14
        self.cfg["env"]["numObservations"] += 21 if not self.control_type == "osc" else 0
        self.obs_act_rew_init()

        super().__init__(config=self.cfg, rl_device=rl_device, sim_device=sim_device, graphics_device_id=graphics_device_id, headless=headless, virtual_screen_capture=virtual_screen_capture, force_render=force_render)

        # custom init
        self.timeaware_init()
        self.env_extra_init()
        self._warmup_env()
        

    def env_extra_init(self):
        # self.franka_default_dof_pos = to_torch(
        #     [0, 0, 0, -2.3180, 0, 2.4416, -0.7854, 0.04, 0.04], device=self.device
        # )
        # Setup focus for velocity tracking
        self.focus_names = ["drawer"]
        self.focus_linvel_names = [f"{focus_name}_linvel_norm" for focus_name in self.focus_names]
        self.focus_linacc_names = [f"{focus_name}_linacc_norm" for focus_name in self.focus_names]
        self.apply_force_handle = "drawer_handle"

        # Used to record the initial drawer handle position and orientation for dof computation
        self._init_pos_get_reset_buf = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.long)
        self.init_drawer_handle_pos = torch.zeros(
            (self.num_envs, 3), device=self.device, dtype=torch.float)
        self.init_drawer_handle_quat = torch.zeros(
            (self.num_envs, 4), device=self.device, dtype=torch.float)


    def create_sim(self):
        self.sim_params.up_axis = gymapi.UP_AXIS_Z
        self.sim_params.gravity.x = 0
        self.sim_params.gravity.y = 0
        self.sim_params.gravity.z = -9.81
        self.sim = super().create_sim(
            self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        self._create_ground_plane()
        self._create_envs(self.num_envs, self.cfg["env"]['envSpacing'], int(np.sqrt(self.num_envs)))

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        # self.gym.add_ground(self.sim, plane_params)


    def _create_cabinet_asset(self, asset_root):
        # load cabinet asset
        cabinet_asset_file = self.cfg["env"]["asset"]["assetFileNameCabinet"]
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        asset_options.collapse_fixed_joints = True
        asset_options.disable_gravity = False
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        asset_options.armature = 0.005
        cabinet_asset = self.gym.load_asset(self.sim, asset_root, cabinet_asset_file, asset_options)
        self.num_cabinet_bodies = self.gym.get_asset_rigid_body_count(cabinet_asset)
        self.num_cabinet_dofs = self.gym.get_asset_dof_count(cabinet_asset)
        print("num cabinet bodies: ", self.num_cabinet_bodies)
        print("num cabinet dofs: ", self.num_cabinet_dofs)

        # set cabinet dof properties
        cabinet_dof_props = self.gym.get_asset_dof_properties(cabinet_asset)
        for i in range(self.num_cabinet_dofs): # friction 1.2 or 1.3 seems better
            cabinet_dof_props['damping'][i] = 50
            cabinet_dof_props['friction'][i] = 1.4 * self.cfg.get("friction_mul", 1)

        # create prop assets
        prop_size = [0.05] * 3
        prop_spacing = 0.09
        prop_opts = gymapi.AssetOptions()
        prop_opts.density = 500 # dont know why but changing the density does not change any behavior
        prop_asset = self.gym.create_box(self.sim, *prop_size, prop_opts)

        return cabinet_asset, cabinet_dof_props, prop_asset, prop_spacing


    def _create_envs(self, num_envs, spacing, num_per_row):
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../assets")
        # Create franka asset
        franka_asset, franka_dof_props = self._create_franka_assets(asset_root)
        cabinet_asset, self.cabinet_dof_props, prop_asset, prop_spacing = self._create_cabinet_asset(asset_root)

        # Create table asset
        table_pos = [0.0, 0.0, -1000]
        table_thickness = 0.05
        table_opts = gymapi.AssetOptions()
        table_opts.fix_base_link = True
        table_asset = self.gym.create_box(self.sim, *[1.2, 1.2, table_thickness], table_opts)
        table_props = self.gym.get_asset_rigid_shape_properties(table_asset)[0]
        self.gym.set_asset_rigid_shape_properties(table_asset, [table_props])
        table_color = gymapi.Vec3(160/255, 160/255, 1.)

        # Create table stand asset
        table_stand_thickness = 0.013
        table_stand_pos = [-0.5, 0.0, 1.0 + table_thickness / 2 + table_stand_thickness / 2]
        table_stand_opts = gymapi.AssetOptions()
        table_stand_opts.fix_base_link = True
        table_stand_asset = self.gym.create_box(self.sim, *[0.2, 0.2, table_stand_thickness], table_stand_opts)

        # Create Workspace stand asset
        ws_stand_thickness = 0.00001
        ws_stand_pos = [0.035, 0.0, 1.0 + table_thickness / 2 + ws_stand_thickness / 2]
        ws_stand_opts = gymapi.AssetOptions()
        ws_stand_opts.fix_base_link = True
        ws_stand_asset = self.gym.create_box(self.sim, *[0.75, 0.75, ws_stand_thickness], ws_stand_opts)

        # Make sure the franka initial quat is same as the world!
        # There is a weird bug of isaacgym where if the initial orientation is not same
        # The osc controller velocity part after conversion will get larger and larger
        franka_start_pose = gymapi.Transform()
        franka_start_pose.p = gymapi.Vec3(-0.45, 0.0, 1.0 + table_thickness / 2 + table_stand_thickness)
        franka_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        # Define start pose for table
        table_start_pose = gymapi.Transform()
        table_start_pose.p = gymapi.Vec3(*table_pos)
        table_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        # Define start pose for table stand
        table_stand_start_pose = gymapi.Transform()
        table_stand_start_pose.p = gymapi.Vec3(*table_stand_pos)
        table_stand_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        # Define start pose for ws stand
        ws_stand_start_pose = gymapi.Transform()
        ws_stand_start_pose.p = gymapi.Vec3(*ws_stand_pos)
        ws_stand_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
        self._ws_surface_pos = np.array(ws_stand_pos) + np.array([0, 0, ws_stand_thickness / 2])
        self.reward_settings["ws_surface_height"] = self._ws_surface_pos[2]

        # Define start pose for cabinet stand
        cabinet_start_pose = gymapi.Transform()
        cabinet_start_pose.p = gymapi.Vec3(0.6, 0.0, 0.25 + self._ws_surface_pos[2])
        cabinet_start_pose.r = gymapi.Quat(0.0, 0.0, 1.0, 0.0)
        self.cab_default_state = to_torch([cabinet_start_pose.p.x, cabinet_start_pose.p.y, cabinet_start_pose.p.z,
                                          cabinet_start_pose.r.x, cabinet_start_pose.r.y, cabinet_start_pose.r.z, cabinet_start_pose.r.w,
                                          0.0, 0.0, 0.0, 0.0, 0.0, 0.0], device=self.device)
        init_distance = cabinet_start_pose.p.x - franka_start_pose.p.x
        assert 0.8 < init_distance < 1.2 , f"The distance between the cabinet and the franka should between [0.5, 1.2], but get {init_distance}."

        # compute aggregate size
        num_franka_bodies = self.gym.get_asset_rigid_body_count(franka_asset)
        num_franka_shapes = self.gym.get_asset_rigid_shape_count(franka_asset)
        num_cabinet_bodies = self.gym.get_asset_rigid_body_count(cabinet_asset)
        num_cabinet_shapes = self.gym.get_asset_rigid_shape_count(cabinet_asset)
        num_prop_bodies = self.gym.get_asset_rigid_body_count(prop_asset)
        num_prop_shapes = self.gym.get_asset_rigid_shape_count(prop_asset)
        max_agg_bodies = num_franka_bodies + num_cabinet_bodies + 3 + self.num_props * num_prop_bodies # 2 for table, table stand, and ws stand
        max_agg_shapes = num_franka_shapes + num_cabinet_shapes + 3 + self.num_props * num_prop_shapes # 3 for table, table stand, and ws stand

        self.envs = []
        self.frankas = []
        self.cabinets = []
        self.prop_start = []
        self.drawer2prop_default_pose = []
        

        for i in range(self.num_envs):
            # create env instance
            env_ptr = self.gym.create_env(
                self.sim, lower, upper, num_per_row
            )

            if self.aggregate_mode >= 3:
                self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

            self._franka_id = franka_actor = self.gym.create_actor(env_ptr, franka_asset, franka_start_pose, "franka", i, 0, 0)
            self.gym.set_actor_dof_properties(env_ptr, franka_actor, franka_dof_props)

            if self.aggregate_mode == 2:
                self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

            # Create table
            self._table_id = table_actor = self.gym.create_actor(env_ptr, table_asset, table_start_pose, "table", i, 1, 0)
            table_stand_actor = self.gym.create_actor(env_ptr, table_stand_asset, table_stand_start_pose, "table_stand", i, 1, 0)
            # ws_stand_actor = self.gym.create_actor(env_ptr, ws_stand_asset, ws_stand_start_pose, "ws_stand", i, 1, 0)
            # Set colors
            self.gym.set_rigid_body_color(env_ptr, table_actor, 0, gymapi.MESH_VISUAL, table_color)

            # Create Cabinet
            self._cabinet_id = cabinet_actor = self.gym.create_actor(env_ptr, cabinet_asset, cabinet_start_pose, "cabinet", i, 1, 0)
            self.gym.set_actor_dof_properties(env_ptr, cabinet_actor, self.cabinet_dof_props)
            drawer_link = self.gym.find_actor_rigid_body_handle(env_ptr, cabinet_actor, "drawer_top")
            drawer_pose = self.gym.get_rigid_transform(env_ptr, drawer_link)

            if self.aggregate_mode == 1:
                self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

            # Create props
            if self.num_props > 0:
                self.prop_start.append(self.gym.get_sim_actor_count(self.sim))

                props_per_row = int(np.ceil(np.sqrt(self.num_props)))
                xmin = -0.5 * prop_spacing * (props_per_row - 1)
                yzmin = -0.5 * prop_spacing * (props_per_row - 1)

                prop_count = 0
                for j in range(props_per_row):
                    for k in range(props_per_row):
                        if prop_count >= self.num_props:
                            break
                        propx = xmin + k * prop_spacing
                        propy = yzmin + j * prop_spacing
                        propz = 0
                        prop_state_pose = gymapi.Transform()
                        prop_state_pose.p.x = drawer_pose.p.x + propx
                        prop_state_pose.p.y = drawer_pose.p.y + propy
                        prop_state_pose.p.z = drawer_pose.p.z + propz
                        prop_state_pose.r = gymapi.Quat(0, 0, 0, 1)
                        prop_handle = self.gym.create_actor(env_ptr, prop_asset, prop_state_pose, "prop{}".format(prop_count), i, 0, 0)
                        prop_count += 1

                        prop_idx = j * props_per_row + k
                        self.drawer2prop_default_pose.append([propx, propy, propz,
                                                              prop_state_pose.r.x, prop_state_pose.r.y, prop_state_pose.r.z, prop_state_pose.r.w,
                                                              0, 0, 0, 0, 0, 0])
            
            if self.aggregate_mode > 0:
                self.gym.end_aggregate(env_ptr)

            self.envs.append(env_ptr)
            self.frankas.append(franka_actor)
            self.cabinets.append(cabinet_actor)

        # Compute some useful transformations for resetting
        w2drawer_quat, w2drawer_handle_pos = to_torch([drawer_pose.r.x, drawer_pose.r.y, drawer_pose.r.z, drawer_pose.r.w], device=self.device), to_torch([drawer_pose.p.x, drawer_pose.p.y, drawer_pose.p.z], device=self.device)
        w2cab_quat, w2cab_pos = to_torch([cabinet_start_pose.r.x, cabinet_start_pose.r.y, cabinet_start_pose.r.z, cabinet_start_pose.r.w], device=self.device), to_torch([cabinet_start_pose.p.x, cabinet_start_pose.p.y, cabinet_start_pose.p.z], device=self.device)
        cab2w_quat, cab2w_pos = tf_inverse(w2cab_quat, w2cab_pos)
        cab2drawer_quat, cab2drawer_pos = tf_combine(cab2w_quat, cab2w_pos, w2drawer_quat, w2drawer_handle_pos)
        self.cab2drawer_pose = torch.concatenate([cab2drawer_pos, cab2drawer_quat]).repeat((self.num_envs, 1))
        self.drawer2prop_default_pose = to_torch(self.drawer2prop_default_pose, device=self.device, dtype=torch.float).view(self.num_envs, self.num_props, 13)
        # Setup init state buffer
        self.fk_init_dof = torch.zeros((self.num_envs, self.num_franka_dofs), dtype=torch.float, device=self.device)
        self.cab_base_state = self.cab_default_state.repeat((self.num_envs, 1))
        self.cab_init_dof = torch.zeros((self.num_envs, self.num_cabinet_dofs), dtype=torch.float, device=self.device)
        
        # Initialize data
        self.init_data()

    
    def init_data(self):
        # Setup sim handles
        env_ptr = self.envs[0]
        franka_handle = self._franka_id
        self.common_data_init(env_ptr, franka_handle)
        self.link_handles.update({
            # Cabinet drawer
            "drawer_top": self.gym.find_actor_rigid_body_handle(env_ptr, self._cabinet_id, "drawer_top"),
        })

        for key, handle in self.link_handles.items():
            assert handle != -1, f"Handle {key} not found! Probably you collapse the fixed link or no such link exists."

        # Hard compute the drawer top to drawer hand transform (hardcoded). This might be different for different cabinet models!
        y_axis = 1
        drawer_local_grasp_pose = gymapi.Transform()
        drawer_local_grasp_pose.p = gymapi.Vec3(*get_axis_params(0., y_axis, 0.3))
        drawer_local_grasp_pose.r = gymapi.Quat(0, 0, 0, 1)
        self.drawer2handle_pos = to_torch([drawer_local_grasp_pose.p.x, drawer_local_grasp_pose.p.y,
                                           drawer_local_grasp_pose.p.z], device=self.device).repeat((self.num_envs, 1))
        self.drawer2handle_quat = to_torch([drawer_local_grasp_pose.r.x, drawer_local_grasp_pose.r.y,
                                            drawer_local_grasp_pose.r.z, drawer_local_grasp_pose.r.w], device=self.device).repeat((self.num_envs, 1))
        handle2cab_quat, handle2cab_pos = tf_combine(*tf_inverse(self.drawer2handle_quat, self.drawer2handle_pos),
                                                     *tf_inverse(self.cab2drawer_pose[:, 3:7], self.cab2drawer_pose[:, 0:3]))
        self.handle2cab_pose = torch.cat([handle2cab_pos, handle2cab_quat], dim=-1)
        
        # Define useful orientations
        self.drawer_inward_axis = to_torch([-1, 0, 0], device=self.device).repeat((self.num_envs, 1))
        self.drawer_up_axis = to_torch([0, 0, 1], device=self.device).repeat((self.num_envs, 1))
        self.gripper_forward_axis = to_torch([0, 0, 1], device=self.device).repeat((self.num_envs, 1))
        self.gripper_up_axis = to_torch([0, 1, 0], device=self.device).repeat((self.num_envs, 1))
        
        # Robot base transformation stored for later use
        self._franka_state = self._root_state[:, franka_handle, :]
        # Object Root State
        self._cab_base_state = self._root_state[:, self._cabinet_id]
        if self.num_props > 0:
            self._prop_states = self._root_state[:, -self.num_props:]
        # Drawer state
        self._drawer_state = self._rigid_body_state[:, self.link_handles["drawer_top"], :]
        

    def _update_task_prev_states(self, real=False):
        states, prev_states = self.get_states_dict(real=real)
        for name in ["drawer_handle_pos", "drawer_linvel"]:
            if name not in states: continue
            prev_states[name] = mix_clone(states[name])


    def _update_init_drawer_handle_state(self, drawer_handle_pos, drawer_handle_quat):
        init_env_ids = self._init_pos_get_reset_buf.nonzero().flatten()
        self.init_drawer_handle_pos[init_env_ids] = drawer_handle_pos[init_env_ids]
        self.init_drawer_handle_quat[init_env_ids] = drawer_handle_quat[init_env_ids]
        self._init_pos_get_reset_buf[init_env_ids] = 0
    
    
    def _update_task_states(self):
        # Get drawer position and handle
        w2topDrawer_pos = self._drawer_state[:, 0:3]
        w2topDrawer_quat = self._drawer_state[:, 3:7]
        cabinet_dof_pos = self._q[:, self.num_franka_dofs:]
        cabinet_dof_vel = self._qd[:, self.num_franka_dofs:]
        # Compute actual grasping transforms
        w2drawerHandle_quat, w2drawerHandle_pos = self.apply_drawer2handle_transform(w2topDrawer_quat, w2topDrawer_pos)

        # Convert from the world frame to the robot base frame
        r2drawerHandle_quat, r2drawerHandle_pos = self.point2frankaBase(w2drawerHandle_quat, w2drawerHandle_pos)
        # Reset envs need to record the initial drawer handle position and orientation
        self._update_init_drawer_handle_state(r2drawerHandle_pos, r2drawerHandle_quat)
        
        # Update states
        self.states.update({
            # Cabinet drawer
            "drawer_handle_pos": r2drawerHandle_pos,
            "drawer_handle_quat": r2drawerHandle_quat,
            "init_drawer_handle_pos": self.init_drawer_handle_pos,
            "init_drawer_handle_quat": self.init_drawer_handle_quat,
            "cabinet_dof_pos": cabinet_dof_pos,
            "cabinet_dof_vel": cabinet_dof_vel,
            "drawer_dof_pos": cabinet_dof_pos[:, 3].unsqueeze(-1),
            "drawer_dof_vel": cabinet_dof_vel[:, 3].unsqueeze(-1),
        })

        # ----------------------------------------------------
        # Update states in the world frame for rendering later
        self.world_states.update({
            # Cabinet drawer
            "drawer_handle_pos": w2drawerHandle_pos,
            "drawer_handle_quat": w2drawerHandle_quat,
            "drawer_linvel": self._drawer_state[:, 7:10],
            "drawer_angvel": self._drawer_state[:, 10:13],
        })


    def _update_diff_states(self, real=False):
        # Update previous states to compute acceleration
        # r2drawer_linvel = self.vec2frankaBase(self._drawer_state[:, 7:10])
        # r2drawer_angvel = self.vec2frankaBase(self._drawer_state[:, 10:13])

        states, prev_states = self.get_states_dict(real=real)
        # We use position to compute the linear velocity instead of directly using the linear velocity in isaacgym
        r2drawerHandle_pos = states["drawer_handle_pos"]
        prev_r2drawerHandle_pos = prev_states["drawer_handle_pos"] if "drawer_handle_pos" in prev_states else r2drawerHandle_pos
        r2drawer_linvel = (r2drawerHandle_pos - prev_r2drawerHandle_pos) / self.ctrl_dt
        
        if real:
            prev_r2drawer_linvel = states["drawer_linvel"] if "drawer_linvel" in states else r2drawer_linvel
            r2drawer_linvel = ema_filter(r2drawer_linvel, prev_r2drawer_linvel, alpha=0.9)
        
        states.update({
            "drawer_linvel": r2drawer_linvel,
            "drawer_linvel_norm": mix_norm(r2drawer_linvel, dim=-1, keepdim=True),
        })

        if "drawer_linvel" in states and "drawer_linvel" in prev_states:
            r2cupA_linacc = (states["drawer_linvel"] - prev_states["drawer_linvel"]) / self.ctrl_dt
            states.update({
                "drawer_linacc_norm": mix_norm(r2cupA_linacc, dim=-1, keepdim=True),
            })


    def compute_task_reward(self):
        """
        Watch out! All states info are in the robot base frame!
        """
        # Get current drawer state
        states = self.states
        cabinet_dof_pos = states["cabinet_dof_pos"]
        franka_grasp_pos = states["eef_pos"]
        franka_grasp_rot = states["eef_quat"]
        drawer_grasp_pos = states["drawer_handle_pos"]
        drawer_grasp_rot = states["drawer_handle_quat"]
        franka_lfinger_pos = states["eef_lf_pos"]
        franka_rfinger_pos = states["eef_rf_pos"]
        drawer_linvel_norm = states["drawer_linvel_norm"].flatten()
        gripper_forward_axis = self.gripper_forward_axis
        drawer_inward_axis = self.drawer_inward_axis
        gripper_up_axis = self.gripper_up_axis
        drawer_up_axis = self.drawer_up_axis
        
        # distance from hand to the drawer
        states["approaching_dist"] = d = torch.norm(franka_grasp_pos - drawer_grasp_pos, dim=-1)
        dist_reward = 1 - torch.tanh(10.0 * d)

        r2eef_z_quat = tf_vector(franka_grasp_rot, gripper_forward_axis)
        r2drawer_grasp_forward_quat = tf_vector(drawer_grasp_rot, drawer_inward_axis)
        r2eef_y_quat = tf_vector(franka_grasp_rot, gripper_up_axis)
        r2drawer_grasp_up_quat = tf_vector(drawer_grasp_rot, drawer_up_axis)
        alignment_forward = batch_dot(r2eef_z_quat, r2drawer_grasp_forward_quat).abs()
        alignment_up = batch_dot(r2eef_y_quat, r2drawer_grasp_up_quat).abs()
        rot_reward = 0.5 * (alignment_forward + alignment_up)  # alignment of forward axis for gripper

        is_gripper_approaching = (d<=0.02) & (alignment_forward>0.5) & (alignment_up>0.5)
        is_gripper_closing = self._gripper_mode==1.
        gripper_reward = (is_gripper_closing * is_gripper_approaching) + (0.2 * ~is_gripper_closing * ~is_gripper_approaching)
        dist_reward += gripper_reward

        # reward for opening drawer; This is equivalent to the distance of current pos of drawer handle to the goal pos (z-axis 0.38m). 
        _, _, scale = project_point_on_segment(franka_lfinger_pos, franka_rfinger_pos, drawer_grasp_pos)
        scale = scale.flatten()
        is_handle_between_gripper = (scale>0) & (scale<1)
        is_grasped = is_gripper_approaching & is_handle_between_gripper & is_gripper_closing
        open_reward = cabinet_dof_pos[:, 3] * is_grasped
        
        # regularization on the actions (summed for each environment)
        action_penalty = torch.norm(self.actions, dim=-1)

        # Combine rewards
        reward_settings = self.reward_settings
        rewards = reward_settings["r_dist_scale"] * dist_reward + reward_settings["r_rot_scale"] * rot_reward \
                + reward_settings["r_open_scale"] * open_reward - reward_settings["r_action_penalty_scale"] * action_penalty

        # Success | Reset criterion
        is_drawer_opened = (cabinet_dof_pos[:, 3] > 0.35) * is_grasped
        success_conditions = is_drawer_opened
        violdated_conditions = (self.progress_buf >= self.max_episode_length - 1) | ~is_under_valid_vel(drawer_linvel_norm, self.MAX_VEL_NORM)
        reset_conditions = success_conditions | violdated_conditions

        # We either provide a large success reward or keep the reward as is
        rewards = torch.where(success_conditions, reward_settings["r_success"], rewards)

        # Reset criterion
        reset_buf = torch.where(reset_conditions, torch.ones_like(self.reset_buf), self.reset_buf)
        success_buf = torch.where(success_conditions, torch.ones_like(self.reset_buf), torch.zeros_like(self.reset_buf))
        
        return rewards, reset_buf, success_buf
        

    def get_taskobs_names(self):
        obs_names = ["drawer_handle_pos", "init_drawer_handle_pos", "eef_pos", "eef_quat"]
        obs_names += [] if self.control_type == "osc" else ["q", "prev_tgtq", "prev_dq"]
        obs_names += ["gripper_mode"]
        return obs_names
    

    def task_prim_obs_init(self):
        return
    
    
    def add_priv_taskobs(self, obs_names):
        return obs_names


    def apply_drawer2handle_transform(self, w2topDrawer_quat, w2topDrawer_pos):
        # Get global grasp transforms for drawer
        w2drawerHandle_quat, w2drawerHandle_pos = tf_combine(
            w2topDrawer_quat, w2topDrawer_pos, 
            self.drawer2handle_quat, self.drawer2handle_pos
        )
        return w2drawerHandle_quat, w2drawerHandle_pos


    def _update_props(self, env_ids):
        for env_id in env_ids:
            env_ptr = self.envs[env_id]
            for i in range(self.num_cabinet_dofs):
                self.cabinet_dof_props['friction'][i] = 1.4 * self.cfg.get("friction_mul", 1)
            self.gym.set_actor_dof_properties(env_ptr, self._cabinet_id, self.cabinet_dof_props)
    
    
    def reset_idx(self, env_ids):
        config_index = self.get_config_idx(env_ids)
        new_cfg_env_ids, config_index = self.filter_env_ids(env_ids, config_index)

        # Compute the initial states of all movable objects
        if len(new_cfg_env_ids) > 0:
            # Reset environment configurations and time related states that require speed ratio
            self._reset_timeaware_states(new_cfg_env_ids, config_index)
            if config_index is not None:
                # Load from saved configurations
                init_fk_pos = self.env_configs["franka_state"][config_index].clone()
                cab_base_pose = self.env_configs["cabinet_base_pose"][config_index].clone()
                cab_base_pos, cab_base_ori = cab_base_pose[:, :3], cab_base_pose[:, 3:7]
                cab_init_dof = self.env_configs["cabinet_dof_pos"][config_index].clone()
            else:
                # Update brand new cube states (self._init_cube_state is inside the function)
                cab_base_pos = self.cab_default_state[:3].unsqueeze(0) + self.cur_dr_params["spatial"]["cabinet_pos"] * torch_rand_float(-1, 1, (len(new_cfg_env_ids), 3), device=self.device)
                cab_init_dof = torch.zeros((len(new_cfg_env_ids), self.num_cabinet_dofs), device=self.device)
                
                # Generate new random joint positions (with noise)
                delta_q = self.cur_dr_params["spatial"]["franka_dof"] * torch_rand_float(-1., 1., (len(new_cfg_env_ids), self.num_franka_dofs), device=self.device)
                init_fk_pos = tensor_clamp(
                    self.franka_default_dof_pos.unsqueeze(0) + delta_q,
                    self.franka_dof_lower_limits, self.franka_dof_upper_limits)
                init_fk_pos[:, -2:] = self.franka_default_dof_pos[-1] # The last two joints are always the same

                # Dr the franka controller gains
                self.change_controller_gains(new_cfg_env_ids)

            if not self.training and self.cfg.get("real_robot", False):
                init_fk_pos = tensor_clamp(
                    self.franka_default_dof_pos.unsqueeze(0),
                    self.franka_dof_lower_limits, self.franka_dof_upper_limits)

            self.fk_init_dof[new_cfg_env_ids] = init_fk_pos
            self.cab_base_state[new_cfg_env_ids, :3] = cab_base_pos
            self.cab_init_dof[new_cfg_env_ids] = cab_init_dof
        
        # Set initial DOF states
        self._q[env_ids, :self.num_franka_dofs] = self.fk_init_dof[env_ids, :]
        self._qd[env_ids, :self.num_franka_dofs] = torch.zeros_like(self.fk_init_dof[env_ids, :])
        self._q[env_ids, self.num_franka_dofs:] = self.cab_init_dof[env_ids, :]
        self._qd[env_ids, self.num_franka_dofs:] = torch.zeros_like(self.cab_init_dof[env_ids, :])

        # Set control targets and reset states
        self._pos_control[env_ids, :self.num_franka_dofs] = self.fk_init_dof[env_ids, :]
        self._effort_control[env_ids, :self.num_franka_dofs] = torch.zeros_like(self.fk_init_dof[env_ids, :])

        # Deploy updates via gym API (franka + cabinet)
        multi_env_ids = self._global_indices[env_ids][:, [self._franka_id, self._cabinet_id]].flatten()
        self.gym.set_dof_position_target_tensor_indexed(self.sim,
                                                    gymtorch.unwrap_tensor(self._pos_control),
                                                    gymtorch.unwrap_tensor(multi_env_ids),
                                                    len(multi_env_ids))
        
        self.gym.set_dof_actuation_force_tensor_indexed(self.sim,
                                                    gymtorch.unwrap_tensor(self._effort_control),
                                                    gymtorch.unwrap_tensor(multi_env_ids),
                                                    len(multi_env_ids))
        
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                            gymtorch.unwrap_tensor(self._dof_state),
                                            gymtorch.unwrap_tensor(multi_env_ids),
                                            len(multi_env_ids))
        
        # Deploy movable updates; all movable objects are loaded since the cupA
        obj_root_ids = self._global_indices[env_ids, self._cabinet_id]
        self._cab_base_state[env_ids] = self.cab_base_state[env_ids]
        if self.num_props > 0: # Reset prop states into the drawer
            obj_root_ids = torch.concatenate([obj_root_ids, self._global_indices[env_ids, -self.num_props:].flatten()])
            cab_drawer_quat, cab_drawer_pos = tf_combine(self.cab_base_state[env_ids, 3:7], self.cab_base_state[env_ids, :3],
                                                         self.cab2drawer_pose[env_ids, 3:7], self.cab2drawer_pose[env_ids, :3])
            w2prop_quat, w2prop_pos = tf_combine(cab_drawer_quat.unsqueeze(1).repeat(1, self.num_props, 1),
                                                 cab_drawer_pos.unsqueeze(1).repeat(1, self.num_props, 1),
                                                 self.drawer2prop_default_pose[env_ids, :, 3:7],
                                                 self.drawer2prop_default_pose[env_ids, :, :3])
            self._prop_states[env_ids, :, 0:3] = w2prop_pos
            self._prop_states[env_ids, :, 3:7] = w2prop_quat
            self._prop_states[env_ids, :, 7:] = self.drawer2prop_default_pose[env_ids, :, 7:]
        
        self.gym.set_actor_root_state_tensor_indexed(self.sim, 
                                                    gymtorch.unwrap_tensor(self._root_state),
                                                    gymtorch.unwrap_tensor(obj_root_ids), 
                                                    len(obj_root_ids))

        # Reset buffers
        self._reset_bufs(env_ids)

        # Record the initial configurations if needed
        if self.is_ready_to_record(new_cfg_env_ids):
            self.record_init_configs(new_cfg_env_ids)

        # Need one step to refresh the dof states; Otherwise, the states (such as eef pose) will be outdated
        self.gym.simulate(self.sim)
        if self.device == 'cpu':
            self.gym.fetch_results(self.sim, True)
        if self.force_render:
            self.render()


    def _reset_bufs(self, env_ids):
        self._init_pos_get_reset_buf[env_ids] = 1
        return super()._reset_bufs(env_ids)


    def reset_sim_cab_states(self, cab_pose):
        # Reset the cabinet states and drawer init dof to the given pose
        self._cab_base_state[:, :3] = cab_pose[:, :3]
        self._cab_base_state[:, 3:7] = cab_pose[:, 3:7]
        self._q[:, self.num_franka_dofs:] = self.cab_init_dof

        # Reset the sim states
        multi_obj_ids = cab_ids = self._global_indices[:, self._cabinet_id]
        self.gym.set_actor_root_state_tensor_indexed(self.sim, 
                                                     gymtorch.unwrap_tensor(self._root_state),
                                                     gymtorch.unwrap_tensor(multi_obj_ids), 
                                                     len(multi_obj_ids))
        # Need one step to refresh the dof states; Otherwise, the states (such as eef pose) will be outdated
        self.gym.simulate(self.sim)
        if self.device == 'cpu':
            self.gym.fetch_results(self.sim, True)
        if self.force_render:
            self.render()


    def _init_task_configs_buf(self):
        init_configs = self.extras["init_configs"]
        init_configs["cabinet_base_pose"] = []
        init_configs["cabinet_dof_pos"] = []
        init_configs["franka_state"] = []

    
    def _record_task_init_configs(self, env_ids):
        init_configs = self.extras["init_configs"]
        init_configs["cabinet_dof_pos"].extend(self._q[env_ids, self.num_franka_dofs:].cpu().tolist())
        init_configs["cabinet_base_pose"].extend(self.cab_base_state[env_ids, :7].cpu().tolist())
        init_configs["franka_state"].extend(self._q[env_ids, :self.num_franka_dofs].cpu().tolist())


    def debug_viz(self):
        self.gym.clear_lines(self.viewer)

        # Visualize key points (end effector, drawer, etc.)
        for i in range(self.num_envs):
            # Draw lines representing important axes
            drawer_handle_pos = self.world_states["drawer_handle_pos"][i]
            drawer_rot = self.world_states["drawer_handle_quat"][i]
            grasp_pos = self.world_states["eef_pos"][i]
            grasp_rot = self.world_states["eef_quat"][i]
            eef_pos = self.world_states["eef_pos"][i]
            eef_rot = self.world_states["eef_quat"][i]
            
            # Draw local coordinate frames
            for pos, rot in [(eef_pos, eef_rot), (drawer_handle_pos, drawer_rot), (grasp_pos, grasp_rot)]:
                px = (pos + quat_apply(rot, to_torch([0.1, 0, 0], device=self.device))).cpu().numpy()
                py = (pos + quat_apply(rot, to_torch([0, 0.1, 0], device=self.device))).cpu().numpy()
                pz = (pos + quat_apply(rot, to_torch([0, 0, 0.1], device=self.device))).cpu().numpy()
                
                pos = pos.cpu().numpy()
                self.gym.add_lines(self.viewer, self.envs[i], 1, [pos[0], pos[1], pos[2], px[0], px[1], px[2]], [0.85, 0.1, 0.1])
                self.gym.add_lines(self.viewer, self.envs[i], 1, [pos[0], pos[1], pos[2], py[0], py[1], py[2]], [0.1, 0.85, 0.1])
                self.gym.add_lines(self.viewer, self.envs[i], 1, [pos[0], pos[1], pos[2], pz[0], pz[1], pz[2]], [0.1, 0.1, 0.85])


    #######################################
    ######### RealRobot Related ###########
    #######################################
    def _init_simulation_mode(self, drawer_pose, franka_states):
        """Initialize simulation mode with real world cube poses."""
        # Convert poses to torch tensors
        drawer_handle_quat, drawer_handle_pos = [to_torch(x, device=self.device) for x in drawer_pose]
        
        # Get robot frame transformation
        w2franka_quat = self._franka_state[0, 3:7]
        w2franka_pos = self._franka_state[0, :3]
        
        # Transform cube poses to world frame
        w2drawer_handle_quat, w2drawer_handle_pos = tf_combine(w2franka_quat, w2franka_pos, 
                                                               drawer_handle_quat, drawer_handle_pos)
        w2cab_quat, w2cab_pos = tf_combine(w2drawer_handle_quat, w2drawer_handle_pos, 
                                           self.handle2cab_pose[0, 3:7], self.handle2cab_pose[0, :3])
        # Reset cube states
        w2cab_pose = torch.cat([w2cab_pos, w2cab_quat], dim=-1).unsqueeze(0)
        self.reset_sim_cab_states(w2cab_pose)
        
        # Validate robot state consistency if using FK replay
        if self.cfg.get("use_fk_replay") and franka_states is not None:
            self._validate_robot_state_consistency(franka_states)
        
        self._update_states()
        if hasattr(self, "env_configs") and self.env_configs is not None:
            estimated_time = self._estimate_minimum_time2end_real(self.env_configs, self.states)
            self._reset_timeaware_states_real(estimated_time)
        self.compute_observations()
        return self.obs_buf, self.extras
    
    
    def _get_drawer_pose_with_retry(self, state_estimator, max_attempts=1, init=False):
        """Get drawer pose with retry logic."""
        drawer_pose_outdated = False
        for _ in range(max_attempts):
            state_estimator.update()
            # state_estimator.render(draw=True)
            drawer_pose = state_estimator.get_target_pose()
            if drawer_pose is not None:
                drawer_pose = (to_torch([0, 0, 1., 0]), to_torch([*drawer_pose[1][:2], 0.58-0.013])) # default orientation
                return drawer_pose, drawer_pose_outdated
        
        if init:
            raise ValueError(f"Failed to get drawer pose after {max_attempts} attempts at initialization.")

        if drawer_pose is None:
            drawer_pose = state_estimator.get_drawer_last_pose()
            drawer_pose_outdated = True
            drawer_pose = (to_torch([0, 0, 1., 0]), to_torch([*drawer_pose[1][:2], 0.58-0.013]))
        return drawer_pose, drawer_pose_outdated


    def init_real2sim(self, state_estimator, franka_arm):
        """Initialize the real-to-sim synchronization."""
        # Misc value initialize
        self.occlude_eef2drawer = None
        self.first_occlude_eef2drawer = None

        # Initial states computation
        drawer_pose, _ = self._get_drawer_pose_with_retry(state_estimator, max_attempts=10, init=True)
        franka_states = self._get_robot_state_with_retry(franka_arm, max_attempts=500)
        
        if self.cfg.get("use_sim_pure", False):
            return self._init_simulation_mode(drawer_pose, franka_states)
        else:
            return self._init_real_robot_mode(state_estimator, franka_arm)


    def _update_task_states_real(self, state_estimator):
        """Update task-specific states (drawer position)."""
        drawer_pose, drawer_pose_outdated = self._get_drawer_pose_with_retry(state_estimator)
        
        # Unpack poses
        drawer_handle_quat, drawer_handle_pos = drawer_pose
        
        # Calculate cabinet pose from drawer pose
        cab_quat, cab_pos = tf_combine(drawer_handle_quat, drawer_handle_pos, 
                                       *tf_inverse(self.cab2drawer_pose[0, 3:7], 
                                                   self.cab2drawer_pose[0, :3]))
        
        self._update_init_drawer_handle_state(drawer_handle_pos.unsqueeze(0), 
                                              drawer_handle_quat.unsqueeze(0))
        
        # Compensation if the drawer gets pushed back at the initial contact
        if drawer_handle_pos[0] > self.init_drawer_handle_pos[0, 0]:
            self.init_drawer_handle_pos[:, 0] = drawer_handle_pos[0] + 0.01
            self.init_drawer_handle_pos[:, 1] = drawer_handle_pos[1]
        
        return {
            "drawer_handle_pos": drawer_handle_pos.cpu().numpy(),
            "drawer_handle_quat": drawer_handle_quat.cpu().numpy(),
            "drawer_pose_outdated": drawer_pose_outdated,
            "init_drawer_handle_pos": self.init_drawer_handle_pos[0, :].cpu().numpy(),
            "init_drawer_handle_quat": self.init_drawer_handle_quat[0, :].cpu().numpy(),
            
            "cabinet_base_pos": cab_pos,
            "cabinet_base_quat": cab_quat,
        }


    def _compensate_task_states_real(self, task_states, robot_states):
        # Compensate task states especially occlusion
        if not self.cfg.get("compensate_occlusion", False):
            return task_states

        if task_states["drawer_pose_outdated"]:
            gripper_closing = self._gripper_mode_temp[0]==1

            if not gripper_closing: # Open gripper, clean the previous transformation and we don't care
                self.first_occlude_eef2drawer = None
            
            elif gripper_closing: # Use last visible transformation to compensate
                eef_quat, eef_pos = robot_states["eef_quat"], robot_states["eef_pos"]

                if self.first_occlude_eef2drawer is None: # First initialize
                    occlude_eef2drawer_quat, occlude_eef2drawer_pos = deepcopy(self.occlude_eef2drawer)
                    # For drawer, we might want to constrain the compensation differently
                    occlude_eef2drawer_pos = np.clip(occlude_eef2drawer_pos, -0.02, 0.02) # Adjust based on drawer handle size
                    self.first_occlude_eef2drawer = (occlude_eef2drawer_quat, occlude_eef2drawer_pos)

                print(f"\n Lost Pose At this Moment!! Gripper mode is: {gripper_closing} | first_occlude_eef2drawer is : {self.first_occlude_eef2drawer} \n")
                
                if self.first_occlude_eef2drawer is not None: # Compensate
                    print(f"\n Compensate drawer pose | gripper_closing: {gripper_closing} \n")
                    new_drawer_quat, new_drawer_pos = tf_combine_np(eef_quat, eef_pos, *self.first_occlude_eef2drawer)

                    task_states["drawer_handle_quat"] = new_drawer_quat
                    task_states["drawer_handle_pos"] = new_drawer_pos

        else:
            # Keep updating the transformation when drawer is visible
            drawer_handle_quat, drawer_handle_pos = task_states["drawer_handle_quat"], task_states["drawer_handle_pos"]
            eef_quat, eef_pos = robot_states["eef_quat"], robot_states["eef_pos"]
            self.occlude_eef2drawer = tf_combine_np(*tf_inverse_np(eef_quat, eef_pos), drawer_handle_quat, drawer_handle_pos)

        return task_states


    def _update_debug_info_real(self):
        """Update debug information from real robot states."""
        debug_info = {
            "eef_pos": self.states_real["eef_pos"],
            "eef_quat": self.states_real["eef_quat"],
            "drawer_handle_pos": self.states_real["drawer_handle_pos"],
            "joint_gripper_poss": self.states_real["q_gripper"],
        }
        
        for key, value in debug_info.items():
            self.update_debug_info(key, value)


    def map_real2sim(self):
        """Draw drawer and cabinet poses in simulation visualization."""
        if not self.cfg.get("use_sim_pure", False):
            return
        
        # Get world frame poses
        poses = {
            "drawer": {
                "pos": self.world_states["drawer_handle_pos"][0].cpu().numpy(),
                "quat": self.world_states["drawer_handle_quat"][0].cpu().numpy()
            },
            "cabinet": {
                "pos": self.world_states["cabinet_base_pos"][0].cpu().numpy(),
                "quat": self.world_states["cabinet_base_quat"][0].cpu().numpy()
            }
        }
        
        # Draw coordinate axes for drawer and cabinet
        self.clear_lines()
        for obj_name, pose in poses.items():
            self.draw_axes(pos=pose["pos"], ori=pose["quat"])


    #-------------Estimate the minimum time for the real new config with KNN-------------# 
    def _compute_init_config_features(self, data_dict):
        # Features: drawer_handle_pos(3), cab_pos(3), franka_dof(7)
        # Note: You might want to include drawer opening amount as a feature
        cabinet_base_pos, cabinet_base_quat = data_dict["cabinet_base_pose"][:, :3], data_dict["cabinet_base_pose"][:, 3:7]
        drawer_handle_quat, drawer_handle_pos = tf_combine(cabinet_base_quat, cabinet_base_pos,
                                                           self.cab2drawer_pose[0, 3:7].unsqueeze(0).repeat((data_dict["cabinet_base_pose"].shape[0], 1)),
                                                           self.cab2drawer_pose[0, :3].unsqueeze(0).repeat((data_dict["cabinet_base_pose"].shape[0], 1)))
        features = torch.cat([drawer_handle_pos, 
                              data_dict["franka_state"][:, :7]], dim=-1)
        return features


    def _time_related_state_names(self):
        return ["drawer_handle_pos", "q"]