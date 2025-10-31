import numpy as np
import os
import torch
import json
import time

from isaacgym import gymtorch
from isaacgym import gymapi

from envs.isaacgymenvs.utils.torch_jit_utils import quat_mul, to_torch, tensor_clamp, quat_apply, torch_rand_float, tf_apply, tf_combine, axisangle2quat
from envs.isaacgymenvs.tasks.base.vec_task import VecTask, is_under_valid_vel, is_under_valid_contact, ema_filter, mix_norm, mix_clone
from tf_utils import tf_inverse as tf_inverse_np, tf_combine as tf_combine_np
from envs.isaacgymenvs.tasks.utils.object_utils import create_hollow_cylinder
from copy import deepcopy


class FrankaCubeStack(VecTask):
    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):
        self.cfg = cfg

        # Custom Variable Init
        self.custom_variable_init()

        # Create dicts to pass to reward function
        self.reward_settings = {
            "r_dist_scale": self.cfg["env"]["distRewardScale"],
            "r_lift_scale": self.cfg["env"]["liftRewardScale"],
            "r_align_scale": self.cfg["env"]["alignRewardScale"],
        }

        self.dr_settings["noise"].update({
            # pos in meters, quat/dof in radians
            "cubeA_pos": [0, 0.01], # in all directions
            "cubeA_quat": [0, np.pi/60], # 3 degrees
            "cubeA_to_cubeB_pos": [0, 0.01], # in all directions
        })

        self.dr_settings["spatial"].update({
            "cubeA_pos": [0., self.start_position_noise],
            "cubeA_quat": [0., self.start_rotation_noise],
            "cubeB_pos": [0., self.start_position_noise],
            "cubeB_quat": [0., self.start_rotation_noise],
        })

        # Dimensions
        # obs include: cubeA_pos (7) + cubeA_to_B_pos (3) + eef_pose (7) + gripper_mode (1) + [q (7) + prev_tgtq (7) + prev_dq (7)] 
        self.cfg["env"]["numObservations"] = 18
        self.cfg["env"]["numObservations"] += 21 if not self.control_type == "osc" else 0
        self.obs_act_rew_init()

        # Env values to be filled in at runtime
        self._init_cubeA_state = None           # Initial state of cubeA for the current env
        self._init_cubeB_state = None           # Initial state of cubeB for the current env
        self._cubeA_state = None                # Current state of cubeA for the current env
        self._cubeB_state = None                # Current state of cubeB for the current env
        self._cubeA_id = None                   # Actor ID corresponding to cubeA for a given env
        self._cubeB_id = None                   # Actor ID corresponding to cubeB for a given env

        super().__init__(config=self.cfg, rl_device=rl_device, sim_device=sim_device, graphics_device_id=graphics_device_id, headless=headless, virtual_screen_capture=virtual_screen_capture, force_render=force_render)

        # custom init
        self.timeaware_init()
        self.env_extra_init()
        self._warmup_env()


    def env_extra_init(self):
        self.focus_names = ["cubeA", "cubeB"]
        self.focus_linvel_names = [f"{focus_name}_linvel_norm" for focus_name in self.focus_names]
        self.focus_linacc_names = [f"{focus_name}_linacc_norm" for focus_name in self.focus_names]
        self.apply_force_handle = "cubeA_handle"

        self.cubeA2grasp_lf = torch.tensor([1, 0, 0.], device=self.device).unsqueeze(0).repeat(self.num_envs, 1) # Add the recommend grasp position


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

    def _create_envs(self, num_envs, spacing, num_per_row):
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../assets")
        # Create franka asset
        franka_asset, franka_dof_props = self._create_franka_assets(asset_root)

        # Create table asset
        table_pos = [0.0, 0.0, 1.0]
        table_thickness = 0.05
        table_opts = gymapi.AssetOptions()
        table_opts.fix_base_link = True
        table_asset = self.gym.create_box(self.sim, *[1.2, 1.2, table_thickness], table_opts)
        table_color = gymapi.Vec3(160/255, 160/255, 1.)

        # Create table stand asset
        table_stand_thickness = 0.013
        table_stand_pos = [-0.5, 0.0, 1.0 + table_thickness / 2 + table_stand_thickness / 2]
        table_stand_opts = gymapi.AssetOptions()
        table_stand_opts.fix_base_link = True
        table_stand_asset = self.gym.create_box(self.sim, *[0.2, 0.2, table_stand_thickness], table_stand_opts)

        # Create Workspace stand asset (real ws has 0.76m length)
        ws_stand_thickness = 0.00001
        ws_stand_pos = [0.035, 0.0, 1.0 + table_thickness / 2 + ws_stand_thickness / 2]
        ws_stand_opts = gymapi.AssetOptions()
        ws_stand_opts.fix_base_link = True
        ws_stand_asset = self.gym.create_box(self.sim, *[0.75, 0.75, ws_stand_thickness], ws_stand_opts)

        self.cubeA_size = 0.050
        self.cubeB_size = 0.070

        # Create cubeA asset
        cubeA_opts = gymapi.AssetOptions()
        cubeA_asset = self.gym.create_box(self.sim, *([self.cubeA_size] * 3), cubeA_opts)
        cubeA_color = gymapi.Vec3(0.6, 0.1, 0.0)

        # Create cubeB asset
        cubeB_opts = gymapi.AssetOptions()
        cubeB_color = gymapi.Vec3(0.0, 0.4, 0.1)
        cubeB_asset = self.gym.create_box(self.sim, *([self.cubeB_size] * 3), cubeB_opts)

        if self.cfg.get("use_container", False): # Overwrite cubeB asset to be a hollow cylinder
            container_asset_fpath = create_hollow_cylinder(
            name="container",
            size=[0.05, 0.07],
            thickness=0.005,
            mass=0.05,
            n_slices=4,
            shape="square",
            use_lid=False,
            transparent_walls=False,
            generate_urdf=True,
            unique_urdf_name=False,
            asset_root_path=asset_root
        )
            cubeB_asset = self.gym.load_asset(self.sim, asset_root, container_asset_fpath, cubeB_opts)
            cubeB_props = self.gym.get_asset_rigid_shape_properties(cubeB_asset)[0]
            # Not sure about how much effect will apply on the contact_offset and rest_offset and why thickness matters
            cubeB_props.friction = 1.0
            cubeB_props.contact_offset = -1
            cubeB_props.thickness = 0.001
            self.gym.set_asset_rigid_shape_properties(cubeB_asset, [cubeB_props])
        
        restitution = 0.5 if self.cfg.get("add_restitution", False) else 0.1
        # Modify dynamics of cubes
        cubeA_props = self.gym.get_asset_rigid_shape_properties(cubeA_asset)
        cubeB_props = self.gym.get_asset_rigid_shape_properties(cubeB_asset)
        cubeA_props[0].restitution = restitution
        cubeB_props[0].restitution = restitution
        self.gym.set_asset_rigid_shape_properties(cubeA_asset, cubeA_props)
        self.gym.set_asset_rigid_shape_properties(cubeB_asset, cubeB_props)

        # Define start pose for franka
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
        self.franka2ws_surface_height = self._ws_surface_pos[2] - franka_start_pose.p.z

        # Define start pose for cubes (doesn't really matter since they're get overridden during reset() anyways)
        cubeA_start_pose = gymapi.Transform()
        cubeA_start_pose.p = gymapi.Vec3(-1.0, 0.0, 0.0)
        cubeA_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
        cubeB_start_pose = gymapi.Transform()
        cubeB_start_pose.p = gymapi.Vec3(1.0, 0.0, 0.0)
        cubeB_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        # compute aggregate size
        num_franka_bodies = self.gym.get_asset_rigid_body_count(franka_asset)
        num_franka_shapes = self.gym.get_asset_rigid_shape_count(franka_asset)
        max_agg_bodies = num_franka_bodies + 5     # 1 for table, table stand, ws_stand, cubeA, cubeB
        max_agg_shapes = num_franka_shapes + 5     # 1 for table, table stand, ws_stand, cubeA, cubeB

        self.frankas = []
        self.envs = []

        # Create environments
        for i in range(self.num_envs):
            # create env instance
            env_ptr = self.gym.create_env(self.sim, lower, upper, num_per_row)

            # Create actors and define aggregate group appropriately depending on setting
            # NOTE: franka should ALWAYS be loaded first in sim!
            if self.aggregate_mode >= 3:
                self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

            # Create franka
            # Potentially randomize start pose
            if self.franka_position_noise > 0:
                rand_xy = self.franka_position_noise * (-1. + np.random.rand(2) * 2.0)
                franka_start_pose.p = gymapi.Vec3(franka_start_pose.p.x + rand_xy[0], 
                                                  franka_start_pose.p.y + rand_xy[1],
                                                  franka_start_pose.p.z)
            if self.franka_rotation_noise > 0:
                rand_rot = torch.zeros(1, 3)
                rand_rot[:, -1] = self.franka_rotation_noise * (-1. + np.random.rand() * 2.0)
                new_quat = axisangle2quat(rand_rot).squeeze().numpy().tolist()
                franka_start_pose.r = gymapi.Quat(*new_quat)
            self._franka_id = franka_actor = self.gym.create_actor(env_ptr, franka_asset, franka_start_pose, "franka", i, 0, 0)
            self.gym.set_actor_dof_properties(env_ptr, franka_actor, franka_dof_props)

            if self.aggregate_mode == 2:
                self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

            # Create table
            self._table_id = table_actor = self.gym.create_actor(env_ptr, table_asset, table_start_pose, "table", i, 1, 0)
            table_stand_actor = self.gym.create_actor(env_ptr, table_stand_asset, table_stand_start_pose, "table_stand", i, 1, 0)
            ws_stand_actor = self.gym.create_actor(env_ptr, ws_stand_asset, ws_stand_start_pose, "ws_stand", i, 1, 0)
            # Set colors
            self.gym.set_rigid_body_color(env_ptr, table_actor, 0, gymapi.MESH_VISUAL, table_color)

            if self.aggregate_mode == 1:
                self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

            # Create cubes
            self._cubeA_id = self.gym.create_actor(env_ptr, cubeA_asset, cubeA_start_pose, "cubeA", i, 2, 0)
            self._cubeB_id = self.gym.create_actor(env_ptr, cubeB_asset, cubeB_start_pose, "cubeB", i, 4, 0)
            # Set colors
            self.gym.set_rigid_body_color(env_ptr, self._cubeA_id, 0, gymapi.MESH_VISUAL, cubeA_color)
            self.gym.set_rigid_body_color(env_ptr, self._cubeB_id, 0, gymapi.MESH_VISUAL, cubeB_color)

            if self.aggregate_mode > 0:
                self.gym.end_aggregate(env_ptr)

            # Store the created env pointers
            self.envs.append(env_ptr)
            self.frankas.append(franka_actor)

        # Setup init state buffer
        self._init_cubeA_state = torch.zeros(self.num_envs, 13, device=self.device)
        self._init_cubeB_state = torch.zeros(self.num_envs, 13, device=self.device)
        self.fk_init_dof = torch.zeros((self.num_envs, 9), dtype=torch.float, device=self.device)

        # Setup data
        self.init_data()


    def init_data(self):
        # Setup sim handles
        env_ptr = self.envs[0]
        franka_handle = 0
        self.common_data_init(env_ptr, franka_handle)
        self.link_handles.update({
            # Cubes
            "cubeA_handle": self.gym.find_actor_rigid_body_handle(env_ptr, self._cubeA_id, "box"),
            "cubeB_handle": self.gym.find_actor_rigid_body_handle(env_ptr, self._cubeB_id, "box"),
        })
       
        # Robot base transformation stored for later use
        self._franka_state = self._root_state[:, franka_handle, :]
        self._cubeA_state = self._root_state[:, self._cubeA_id, :]
        self._cubeB_state = self._root_state[:, self._cubeB_id, :]

        # Initialize states
        self.states.update({
            "cubeA_size": torch.ones((self.num_envs, 3), device=self.device) * self.cubeA_size,
            "cubeB_size": torch.ones((self.num_envs, 3), device=self.device) * self.cubeB_size,
        })


    def _update_task_prev_states(self, real=False):
        states, prev_states = self.get_states_dict(real=real)
        for name in ["cubeA_pos", "cubeB_pos", "cubeA_linvel", "cubeB_linvel"]:
            if name not in states: continue
            prev_states[name] = mix_clone(states[name])
    

    def _update_task_states(self, reset_ids=[]):
        cubeA_pos, cubeA_quat = self._cubeA_state[:, :3], self._cubeA_state[:, 3:7]
        cubeB_pos, cubeB_quat = self._cubeB_state[:, :3], self._cubeB_state[:, 3:7]
        cubeA_grasplf_pos = tf_apply(cubeA_quat, cubeA_pos, self.cubeA2grasp_lf)
        r2cubeA_quat, r2cubeA_pos = self.point2frankaBase(cubeA_quat, cubeA_pos)
        r2cubeB_quat, r2cubeB_pos = self.point2frankaBase(cubeB_quat, cubeB_pos)
        _, r2cubeA_grasplf_pos = self.point2frankaBase(self.unit_quat, cubeA_grasplf_pos)
        
        if self.cfg.get("use_container", False): # Avoid the z position of cubeB being too low
            r2cubeB_pos[:, 2] = torch.clip(r2cubeB_pos[:, 2], min=self.cubeB_size/2-0.013)
        
        self.states.update({
            # Cubes
            "cubeA_quat": r2cubeA_quat,
            "cubeA_pos": r2cubeA_pos,
            "cubeB_quat": r2cubeB_quat,
            "cubeB_pos": r2cubeB_pos,
            "cubeA_to_cubeB_pos": r2cubeB_pos - r2cubeA_pos,
            "cubeA_grasplf_pos": r2cubeA_grasplf_pos,
        })

        # ----------------------------------------------------
        # Update states in the world frame for rendering later
        self.world_states.update({
            # Cubes
            "cubeA_quat": self._cubeA_state[:, 3:7],
            "cubeA_pos": self._cubeA_state[:, :3],
            "cubeB_quat": self._cubeB_state[:, 3:7],
            "cubeB_pos": self._cubeB_state[:, :3],
        })


    def _update_diff_states(self, real=False):
        # Update the states that require the computation from both current states and previous states
        # We do not use the linear velocity from isaacgym directly, since it is not accessible in the real
        # r2cubeA_linvel = self.vec2frankaBase(self._cubeA_state[:, 7:10])
        # r2cubeB_linvel = self.vec2frankaBase(self._cubeB_state[:, 7:10])

        states, prev_states = self.get_states_dict(real=real)

        r2cubeA_pos = states["cubeA_pos"]
        r2cubeB_pos = states["cubeB_pos"]
        prev_r2cubeA_pos = prev_states["cubeA_pos"] if "cubeA_pos" in prev_states else r2cubeA_pos
        prev_r2cubeB_pos = prev_states["cubeB_pos"] if "cubeB_pos" in prev_states else r2cubeB_pos
        r2cubeA_linvel = (r2cubeA_pos - prev_r2cubeA_pos) / self.ctrl_dt
        r2cubeB_linvel = (r2cubeB_pos - prev_r2cubeB_pos) / self.ctrl_dt

        # Use ema filter to smooth the noisy linear velocity for real world. 
        # Simulation smoothness is in collecting_observation after applying position noise.
        if real:
            prev_r2cubeA_linvel = states["cubeA_linvel"] if "cubeA_linvel" in states else r2cubeA_linvel
            prev_r2cubeB_linvel = states["cubeB_linvel"] if "cubeB_linvel" in states else r2cubeB_linvel
            r2cubeA_linvel = ema_filter(r2cubeA_linvel, prev_r2cubeA_linvel, alpha=0.9)
            r2cubeB_linvel = ema_filter(r2cubeB_linvel, prev_r2cubeB_linvel, alpha=0.9)
        
        states.update({
            "cubeA_linvel": r2cubeA_linvel,
            "cubeB_linvel": r2cubeB_linvel,
            "cubeA_linvel_norm": mix_norm(r2cubeA_linvel, dim=-1, keepdim=True),
            "cubeB_linvel_norm": mix_norm(r2cubeB_linvel, dim=-1, keepdim=True),
        })
        
        if "cubeA_linvel" in states and "cubeA_linvel" in prev_states:
            r2cubeA_linacc = (states["cubeA_linvel"] - prev_states["cubeA_linvel"]) / self.ctrl_dt
            r2cubeB_linacc = (states["cubeB_linvel"] - prev_states["cubeB_linvel"]) / self.ctrl_dt
            states.update({
                "cubeA_linacc_norm": mix_norm(r2cubeA_linacc, dim=-1, keepdim=True),
                "cubeB_linacc_norm": mix_norm(r2cubeB_linacc, dim=-1, keepdim=True),
            })

        # We use position to compute the linear velocity instead of directly using the linear velocity in isaacgym
        # r2cubeA_pos = self.states["cubeA_pos"]
        # r2cubeB_pos = self.states["cubeB_pos"]
        # prev_r2cubeA_pos = prev_states["cubeA_pos"] if "cubeA_pos" in prev_states else r2cubeA_pos
        # prev_r2cubeB_pos = prev_states["cubeB_pos"] if "cubeB_pos" in prev_states else r2cubeB_pos
        # r2cubeA_linvel = (r2cubeA_pos - prev_r2cubeA_pos) / self.ctrl_dt
        # r2cubeB_linvel = (r2cubeB_pos - prev_r2cubeB_pos) / self.ctrl_dt

        # self.extras.update({
        #     "cubeA_linvel_finite": r2cubeA_linvel,
        #     "cubeB_linvel_finite": r2cubeB_linvel,
        #     "cubeA_linvel_finite_norm": r2cubeA_linvel.norm(dim=-1, keepdim=True),
        #     "cubeB_linvel_finite_norm": r2cubeB_linvel.norm(dim=-1, keepdim=True),
        # })


    def compute_task_reward(self):
        """
        Watch out! All states info are in the robot base frame!
        """
        cfg = self.cfg
        states = self.states
        prev_states = self.prev_states
        reward_settings = self.reward_settings

        # Compute per-env physical parameters
        cubeA_size = states["cubeA_size"][:, 2]
        cubeB_size = states["cubeB_size"][:, 2]
        lift_height = cubeA_size + cubeB_size * 1.5
        target_height = cubeB_size + cubeA_size / 2 if not self.cfg.get("use_container", False) else cubeA_size
        init_height = cubeA_size / 2.0

        # Scene stability
        cubeA_linvel_norm = states["cubeA_linvel_norm"].flatten()
        cubeA_linacc_norm = states["cubeA_linacc_norm"].flatten()
        cubeB_linvel_norm = states["cubeB_linvel_norm"].flatten()
        arm_qvel = states["q_vel"]
        arm_qacc = (arm_qvel - prev_states["q_vel"]) / self.ctrl_dt
        table_contact_forces_norm = torch.norm(states["table_contact_forces"], dim=-1)

        # distance from hand to the cubeA, needs to close the gripper when it is close enough
        states["approaching_dist"] = d = torch.norm(states["cubeA_pos"] - states["eef_pos"], dim=-1)
        d_lf = torch.norm(states["cubeA_pos"] - states["eef_lf_pos"], dim=-1)
        d_rf = torch.norm(states["cubeA_pos"] - states["eef_rf_pos"], dim=-1)
        dist_reward = 0.5 * (1 - torch.tanh(10.0 * d) + (1 - torch.tanh(10.0 * (d_lf + d_rf) / 2)) * (d <= 0.01))

        # reward for lifting cubeA; We assume cubeA is on the table; 
        # Since CubeA is in base frame, we convert it to base bottom frame.
        cubeA_height = states["cubeA_pos"][:, 2] - self.franka2ws_surface_height
        lift_reward = torch.clamp(cubeA_height-init_height, 
                                  min=torch.zeros_like(cubeA_height), 
                                  max=lift_height-init_height)
        cubeA_lifted = cubeA_height >= lift_height

        # how closely aligned cubeA is to cubeB (only provided if cubeA is lifted)
        cubeA_to_target_pos = states["cubeA_to_cubeB_pos"].clone()
        cubeA_to_target_pos[:, 2] += (cubeA_size + cubeB_size) / 2 # CubeB_pos + z_offset - CubeA_pos (target offset is above cubeB)
        d_ab = torch.norm(cubeA_to_target_pos, dim=-1)
        align_reward = (1 - torch.tanh(10.0 * d_ab)) * cubeA_lifted
        cubeA_align_cubeB = torch.norm(states["cubeA_to_cubeB_pos"][:, :2], dim=-1) < 0.02

        # gripper release bonus
        goal_lf = torch.norm(self.franka_dof_upper_limits[-2] - states["q_gripper"][:, 0], dim=-1)
        goal_rf = torch.norm(self.franka_dof_upper_limits[-1] - states["q_gripper"][:, 1], dim=-1)
        release_reward = (1 - torch.tanh(10.0 * (goal_lf + goal_rf))) * cubeA_align_cubeB
        align_reward = (align_reward + release_reward) / 2

        # final reward for stacking successfully (only if cubeA is close to target height and corresponding location, and gripper is not grasping)
        cubeA_on_cubeB = torch.abs(cubeA_height - target_height) < 0.02
        gripper_away_from_cubeA = (d > self.cfg.get("away_dist", 0.06))
        stack_reward = cubeA_align_cubeB & cubeA_on_cubeB & gripper_away_from_cubeA

        # regularization on the actions and forces (summed for each environment)
        action_penalty = torch.norm(self.actions, dim=-1)
        force_penalty = torch.zeros_like(table_contact_forces_norm)
        if reward_settings["r_force_penalty_scale"] > 0: # reduce computation if not needed
            force_penalty = torch.where(table_contact_forces_norm > self.MAX_CONTACT_FORCE_NORM, table_contact_forces_norm, force_penalty)

        # Compose rewards
        # We either provide the stack reward or the align + dist reward
        rewards = torch.where(
            stack_reward,
            reward_settings["r_success"] * stack_reward,
            reward_settings["r_dist_scale"] * dist_reward + reward_settings["r_lift_scale"] * lift_reward + reward_settings["r_align_scale"] * align_reward,
        )

        # We provide constraints at each step
        rewards -= reward_settings["r_action_penalty_scale"] * action_penalty + reward_settings["r_force_penalty_scale"] * force_penalty

        # Compute resets
        success_conditions = (stack_reward > 0) & (cubeA_linvel_norm < 0.05)
        violated_conditions = ~is_under_valid_vel(cubeA_linvel_norm, self.MAX_VEL_NORM) | ~is_under_valid_vel(cubeB_linvel_norm, self.MAX_VEL_NORM) | (self.progress_buf >= self.max_episode_length - 1)
        if self.cur_curri_ratio==1:
            violated_conditions |= ~is_under_valid_contact(table_contact_forces_norm, self.MAX_CONTACT_FORCE_NORM)
        
        reset_conditions = success_conditions | violated_conditions
        reset_buf = torch.where(reset_conditions, torch.ones_like(self.reset_buf), self.reset_buf)
        # reset_buf = torch.where((progress_buf >= max_episode_length - 1), torch.ones_like(reset_buf), reset_buf) # The agent needs to learn to move away from the stacking objects
        # Compute Success
        success_buf = torch.where(success_conditions, torch.ones_like(self.reset_buf), torch.zeros_like(self.reset_buf))
        return rewards, reset_buf, success_buf


    def get_taskobs_names(self):
        obs_names = ["cubeA_pos", "cubeA_quat", "cubeA_to_cubeB_pos", "eef_pos", "eef_quat"]
        obs_names += [] if self.control_type == "osc" else ["q", "prev_tgtq", "prev_dq"]
        obs_names += ["gripper_mode"]
        return obs_names
    

    def task_prim_obs_init(self):
        pass
    
    
    def add_priv_taskobs(self, obs_names):
        return obs_names
    

    def reset_idx(self, env_ids):
        config_index = self.get_config_idx(env_ids)
        new_cfg_env_ids, config_index = self.filter_env_ids(env_ids, config_index)

        # Compute the initial states of all movable objects
        if len(new_cfg_env_ids) > 0:
            # Reset environment configurations and time related states that require speed ratio
            self._reset_timeaware_states(new_cfg_env_ids, config_index)
            if config_index is not None:
                self._init_cubeA_state[new_cfg_env_ids] = self._apply_cube_state_noise(self.env_configs["cubeA_state"][config_index].clone(), new_cfg_env_ids) \
                                                           if (self.training and self.cfg.get("add_cube_noise", False)) else self.env_configs["cubeA_state"][config_index].clone()
                self._init_cubeB_state[new_cfg_env_ids] = self._apply_cube_state_noise(self.env_configs["cubeB_state"][config_index].clone(), new_cfg_env_ids) \
                                                           if (self.training and self.cfg.get("add_cube_noise", False)) else self.env_configs["cubeB_state"][config_index].clone()
                init_fk_pos = self.env_configs["franka_state"][config_index].clone()
            else:
                # Update brand new cube states (self._init_cube_state is inside the function)
                self._init_cubeB_state[new_cfg_env_ids] = self._reset_init_cube_state("cubeB", 
                                                                                      self.states["cubeB_size"], 
                                                                                      self.states["cubeB_size"][:, 2]/2,
                                                                                      self._init_cubeA_state, 
                                                                                      self.states["cubeA_size"], 
                                                                                      env_ids=new_cfg_env_ids, 
                                                                                      check_valid=False)
                self._init_cubeA_state[new_cfg_env_ids] = self._reset_init_cube_state("cubeA", 
                                                                                      self.states["cubeA_size"], 
                                                                                      self.states["cubeA_size"][:, 2]/2,
                                                                                      self._init_cubeB_state, 
                                                                                      self.states["cubeB_size"], 
                                                                                      env_ids=new_cfg_env_ids, 
                                                                                      check_valid=True)
                
                # Reset the franka initial position
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
                
            self.fk_init_dof[new_cfg_env_ids, :] = init_fk_pos

        # Write these new init states to the sim states
        self._cubeA_state[env_ids] = self._init_cubeA_state[env_ids]
        self._cubeB_state[env_ids] = self._init_cubeB_state[env_ids]
        # Reset the internal obs accordingly
        self._q[env_ids, :] = self.fk_init_dof[env_ids, :]
        self._qd[env_ids, :] = torch.zeros_like(self._qd[env_ids])

        # Set any position control to the current position, and any vel / effort control to be 0
        # NOTE: Task takes care of actually propagating these controls in sim using the SimActions API
        self._pos_control[env_ids, :] = self.fk_init_dof[env_ids, :]
        self._effort_control[env_ids, :] = torch.zeros_like(self.fk_init_dof[env_ids, :])

        # Deploy arm updates
        multi_env_ids = self._global_indices[env_ids, 0].flatten()
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

        # Deploy cube updates
        multi_env_ids_cubes_int32 = self._global_indices[env_ids, -2:].flatten()
        self.gym.set_actor_root_state_tensor_indexed(self.sim, 
                                                     gymtorch.unwrap_tensor(self._root_state),
                                                     gymtorch.unwrap_tensor(multi_env_ids_cubes_int32), 
                                                     len(multi_env_ids_cubes_int32))

        # Reset buffers
        self._reset_bufs(env_ids)

        # Record the initial configurations
        if self.is_ready_to_record(new_cfg_env_ids):
            self.record_init_configs(new_cfg_env_ids)

        # Need one step to refresh the dof states; Otherwise, the states (such as eef pose) will be outdated
        self.gym.simulate(self.sim)
        if self.device == 'cpu':
            self.gym.fetch_results(self.sim, True)
        if self.force_render:
            self.render()


    def reset_real_cube_states(self, cubeA_pose, cubeB_pose):
        # Reset the cube states to the given pose
        self._cubeA_state[:, :7] = cubeA_pose[:, :7]
        self._cubeB_state[:, :7] = cubeB_pose[:, :7] 

        # Reset the sim states
        cubeA_ids = self._global_indices[:, self._cubeA_id]
        cubeB_ids = self._global_indices[:, self._cubeB_id]
        multi_obj_ids = torch.cat([cubeA_ids, cubeB_ids], dim=-1)
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
        init_configs["cubeA_state"] = []
        init_configs["cubeB_state"] = []
        init_configs["franka_state"] = []
        init_configs["cubeA2cubeB_dist"] = []
        init_configs["full_dist"] = []


    def _record_task_init_configs(self, env_ids):
        init_configs = self.extras["init_configs"]
        init_configs["cubeA_state"].extend(self._init_cubeA_state[env_ids].cpu().tolist())
        init_configs["cubeB_state"].extend(self._init_cubeB_state[env_ids].cpu().tolist())
        init_configs["franka_state"].extend(self._q[env_ids].cpu().tolist())
        
        cubeA2cubeB_dist = torch.norm(self._init_cubeA_state[env_ids, :3] - self._init_cubeB_state[env_ids, :3], dim=-1)
        eef2cubeA_dist = torch.norm(self._init_cubeA_state[env_ids, :3] - self._eef_state[env_ids, :3], dim=-1)
        init_configs["cubeA2cubeB_dist"].extend(cubeA2cubeB_dist.cpu().tolist())
        init_configs["full_dist"].extend((cubeA2cubeB_dist + eef2cubeA_dist).cpu().tolist())
    
        
    def debug_viz(self):
        # debug viz
        self.gym.clear_lines(self.viewer)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        # Grab relevant states to visualize
        eef_pos = self.states["eef_pos"]
        eef_quat = self.states["eef_quat"]
        cubeA_pos = self.states["cubeA_pos"]
        cubeA_quat = self.states["cubeA_quat"]
        cubeB_pos = self.states["cubeB_pos"]
        cubeB_quat = self.states["cubeB_quat"]

        # print(f"EEf Pose: {eef_pos[0]}")
        # EEF pose: 5.3342e-01, 2.1190e-07, 2.5151e-01
        # Cube Pose: tensor([ 0.4850, -0.0990,  0.0120], device='cuda:0'), Target Pose: tensor([0.4850, 0.0990, 0.0220], device='cuda:0')

        eef_quat, eef_pos = tf_combine(self._franka_state[:, 3:7], self._franka_state[:, :3], eef_quat, eef_pos)
        cubeA_quat, cubeA_pos = tf_combine(self._franka_state[:, 3:7], self._franka_state[:, :3], cubeA_quat, cubeA_pos)
        cubeB_quat, cubeB_pos = tf_combine(self._franka_state[:, 3:7], self._franka_state[:, :3], cubeB_quat, cubeB_pos)

        # Plot visualizations
        for i in range(self.num_envs):
            for pos, rot in zip((eef_pos, cubeA_pos, cubeB_pos), (eef_quat, cubeA_quat, cubeB_quat)):
                px = (pos[i] + quat_apply(rot[i], to_torch([1, 0, 0], device=self.device) * 0.2)).cpu().numpy()
                py = (pos[i] + quat_apply(rot[i], to_torch([0, 1, 0], device=self.device) * 0.2)).cpu().numpy()
                pz = (pos[i] + quat_apply(rot[i], to_torch([0, 0, 1], device=self.device) * 0.2)).cpu().numpy()

                p0 = pos[i].cpu().numpy()
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], px[0], px[1], px[2]], [0.85, 0.1, 0.1])
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], py[0], py[1], py[2]], [0.1, 0.85, 0.1])
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], pz[0], pz[1], pz[2]], [0.1, 0.1, 0.85])


    def draw_arrow(self, start, end, color=(0.85, 0.1, 0.1), length=0.1, thickness=0.01):
        """Draw an arrow from start to end with given color, length, and thickness."""
        self.gym.clear_lines(self.viewer)
        dir_vec = end - start
        dir_norm = np.linalg.norm(dir_vec)
        if dir_norm < 1e-6:
            return


    #######################################
    ######### RealRobot Related ###########
    #######################################
    def _init_simulation_mode(self, cubeA_pose, cubeB_pose, franka_states):
        """Initialize simulation mode with real world cube poses."""
        # Convert poses to torch tensors
        cubeA_quat, cubeA_pos = [to_torch(x, device=self.device) for x in cubeA_pose]
        cubeB_quat, cubeB_pos = [to_torch(x, device=self.device) for x in cubeB_pose]
        
        # Get robot frame transformation
        w2franka_quat = self._franka_state[0, 3:7]
        w2franka_pos = self._franka_state[0, :3]
        
        # Transform cube poses to world frame
        w2cube_quat, w2cube_pos = tf_combine(w2franka_quat, w2franka_pos, cubeA_quat, cubeA_pos)
        w2target_quat, w2target_pos = tf_combine(w2franka_quat, w2franka_pos, cubeB_quat, cubeB_pos)
        w2cube_pos[2] = torch.clamp(w2cube_pos[2], min=self._ws_surface_pos[2] + self.cubeA_size / 2)
        w2target_pos[2] = torch.clamp(w2target_pos[2], min=self._ws_surface_pos[2] + self.cubeB_size / 2)
        
        # Reset cube states
        w2cubeA_pose = torch.cat([w2cube_pos, w2cube_quat], dim=-1).unsqueeze(0)
        w2cubeB_pose = torch.cat([w2target_pos, w2target_quat], dim=-1).unsqueeze(0)
        self.reset_real_cube_states(w2cubeA_pose, w2cubeB_pose)
        
        # Validate robot state consistency if using FK replay
        if self.cfg.get("use_fk_replay") and franka_states is not None:
            self._validate_robot_state_consistency(franka_states)
        
        self._update_states()
        if hasattr(self, "env_configs") and self.env_configs is not None:
            estimated_time = self._estimate_minimum_time2end_real(self.env_configs, self.states)
            self._reset_timeaware_states_real(estimated_time)
        self.compute_observations()
        return self.obs_buf, self.extras
    
    
    def _get_cube_poses_with_retry(self, state_estimator, max_attempts=1, init=False):
        """Get cube poses with retry logic."""
        cubeA_pose_outdated, cubeB_pose_outdated = False, False
        for _ in range(max_attempts):
            state_estimator.update()
            state_estimator.render(draw=True)
            cubeA_pose = state_estimator.get_cube_pose()
            cubeB_pose = state_estimator.get_target_pose()
            
            if cubeA_pose is not None and cubeB_pose is not None:
                return cubeA_pose, cubeA_pose_outdated, cubeB_pose, cubeB_pose_outdated
        
        if init:
            raise ValueError(f"Failed to get cube poses after {max_attempts} attempts at initialization.")

        if cubeA_pose is None:
            cubeA_pose = state_estimator.get_cube_last_pose()
            cubeA_pose_outdated = True

        if cubeB_pose is None:
            cubeB_pose = state_estimator.get_target_last_pose()
            cubeB_pose_outdated = True
        return cubeA_pose, cubeA_pose_outdated, cubeB_pose, cubeB_pose_outdated
    
    
    def init_real2sim(self, state_estimator, franka_arm):
        """Initialize the real-to-sim synchronization."""
        # Misc value initialize
        self.occlude_eef2cubeA = None
        self.first_occlude_eef2cubeA = None

        # Initial states computation
        cubeA_pose, _, cubeB_pose, _ = self._get_cube_poses_with_retry(state_estimator, max_attempts=10, init=True)
        franka_states = self._get_robot_state_with_retry(franka_arm, max_attempts=500)
        
        if self.cfg.get("use_sim_pure", False):
            return self._init_simulation_mode(cubeA_pose, cubeB_pose, franka_states)
        else:
            return self._init_real_robot_mode(state_estimator, franka_arm)


    def _update_task_states_real(self, state_estimator):
        """Update task-specific states (cube positions)."""
        cubeA_pose, cubeA_pose_outdated, cubeB_pose, cubeB_pose_outdated = self._get_cube_poses_with_retry(state_estimator)
        
        # Unpack poses
        cubeA_quat, cubeA_pos = cubeA_pose
        cubeB_quat, cubeB_pos = cubeB_pose
        
        return {
            "cubeA_pos": cubeA_pos,
            "cubeA_quat": cubeA_quat,
            "cubeA_pose_outdated": cubeA_pose_outdated,
            "cubeB_pos": cubeB_pos,
            "cubeB_quat": cubeB_quat,
            "cubeB_pose_outdated": cubeB_pose_outdated,
            "cubeA_to_cubeB_pos": cubeB_pos - cubeA_pos,
        }
    

    def _compensate_task_states_real(self, task_states, robot_states):
        # Compensate task states especially occlusion
        if not self.cfg.get("compensate_occlusion", False):
            return task_states

        if task_states["cubeA_pose_outdated"]:
            gripper_closing = self._gripper_mode_temp[0]==1

            if not gripper_closing: # Open gripper, clean the previous transformation and we don't care
                self.first_occlude_eef2cubeA = None
            
            elif gripper_closing: # Use last visible transformation to compensate
                eef_quat, eef_pos = robot_states["eef_quat"], robot_states["eef_pos"]
                # cubeA_quat, cubeA_pos = task_states["cubeA_quat"], task_states["cubeA_pos"]
                # # eef2cube_distance = np.linalg.norm(cubeA_pos - eef_pos)
                # # eef_close_enough = eef2cube_distance < self.cubeA_size

                if self.first_occlude_eef2cubeA is None: # First initialize
                    occlude_eef2cubeA_quat, occlude_eef2cubeA_pos = deepcopy(self.occlude_eef2cubeA)
                    occlude_eef2cubeA_pos = np.clip(occlude_eef2cubeA_pos, -0.02, 0.02) # Assume the cube is in hand
                    self.first_occlude_eef2cubeA = (occlude_eef2cubeA_quat, occlude_eef2cubeA_pos)

                print(f"\n Lost Pose At this Moment!! Gripper mode is: {gripper_closing} | first_occlude_eef2cubeA is : {self.first_occlude_eef2cubeA} \n")
                
                if self.first_occlude_eef2cubeA is not None: # Compensate
                    print(f"\n Compensate cubeA pose | gripper_closing: {gripper_closing} \n")
                    new_cubeA_quat, new_cubeA_pos = tf_combine_np(eef_quat, eef_pos, *self.first_occlude_eef2cubeA)

                    task_states["cubeA_quat"] = new_cubeA_quat
                    task_states["cubeA_pos"] = new_cubeA_pos
                    task_states["cubeA_to_cubeB_pos"] = task_states["cubeB_pos"] - new_cubeA_pos
        else:
            # Keep updating the transformation when cubeA is visible
            cubeA_quat, cubeA_pos = task_states["cubeA_quat"], task_states["cubeA_pos"]
            eef_quat, eef_pos = robot_states["eef_quat"], robot_states["eef_pos"]
            self.occlude_eef2cubeA = tf_combine_np(*tf_inverse_np(eef_quat, eef_pos), cubeA_quat, cubeA_pos)

        return task_states


    def _update_debug_info_real(self):
        """Update debug information from real robot states."""
        debug_info = {
            "eef_pos": self.states_real["eef_pos"],
            "eef_quat": self.states_real["eef_quat"],
            "cubeA_pos": self.states_real["cubeA_pos"],
            "joint_gripper_poss": self.states_real["q_gripper"],
        }
        
        for key, value in debug_info.items():
            self.update_debug_info(key, value)


    def map_real2sim(self):
        """Draw cube poses in simulation visualization."""
        if not self.cfg.get("use_sim_pure", False):
            return
        
        # Get world frame poses
        poses = {
            "cubeA": {
                "pos": self.world_states["cubeA_pos"][0].cpu().numpy(),
                "quat": self.world_states["cubeA_quat"][0].cpu().numpy()
            },
            "cubeB": {
                "pos": self.world_states["cubeB_pos"][0].cpu().numpy(),
                "quat": self.world_states["cubeB_quat"][0].cpu().numpy()
            }
        }
        
        # Draw coordinate axes for each cube
        self.clear_lines()
        for cube_name, pose in poses.items():
            self.draw_axes(pos=pose["pos"], ori=pose["quat"])


    #-------------Estimate the minimum time for the real new config with KNN-------------# 
    def _compute_init_config_features(self, data_dict):
        # Features: cubeA_pos(3), cubeB_pos(3), franka_dof(7)
        features = torch.cat([data_dict["cubeA_state"][:, :3], 
                              data_dict["cubeB_state"][:, :3], 
                              data_dict["franka_state"][:, :7]], dim=-1)
        return features

    
    def _time_related_state_names(self):
        return ["cubeA_pos", "cubeB_pos", "q"]


#####################################################################
###=========================jit functions=========================###
#####################################################################
# Weird error, the torch.jit.script with rewards seem introduce cycling tensor and introducing gpu memory leak problem
# rewards can not be memorized in any way such as self.rew_buf = rewards, or reward_buf[step] = rewards ... Weird, only to rewards
# However, the rl_games can bypass this problem and keeps using torch.jit.script. Fix later