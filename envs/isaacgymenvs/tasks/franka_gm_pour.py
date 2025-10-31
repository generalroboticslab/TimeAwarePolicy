import numpy as np
import os
import torch
import json

from isaacgym import gymtorch
from isaacgym import gymapi

from envs.isaacgymenvs.utils.torch_jit_utils import quat_mul, to_torch, tensor_clamp, quat_apply, torch_rand_float, tf_apply, tf_inverse, tf_combine, axisangle2quat, normalize, batch_dot
from envs.isaacgymenvs.tasks.base.vec_task import VecTask, is_in_cup, project_point_on_segment, is_under_valid_vel, is_under_valid_contact, mix_clone, mix_norm, ema_filter
from envs.isaacgymenvs.tasks.utils.object_utils import create_hollow_cylinder, create_hollow_cylinder_mesh
from copy import deepcopy
from tf_utils import tf_inverse as tf_inverse_np, tf_combine as tf_combine_np


class FrankaGmPour(VecTask):
    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):
        """
        If you want to add a new environment, you need to: 
        1) add the class in the IsaacGymEnvs/isaacgymenvs/tasks/__init__.py; 
        2) Add yaml in the IsaacGymEnvs/isaacgymenvs/cfg/task with name ENVNAME 
        3) Add yaml in the IsaacGymEnvs/isaacgymenvs/cfg/train with name ENVNAMEPPO

        Remeber, when you create a new urdf, the inertial part of each link is strictly required. Otherwise, the simulation will not work!!!
        """
        self.cfg = cfg

        # Custom
        self.custom_variable_init()

        # Create dicts to pass to reward function
        self.reward_settings = {
            "r_dist_scale": self.cfg["env"]["distRewardScale"],
            "r_lift_scale": self.cfg["env"]["liftRewardScale"],
            "r_align_scale": self.cfg["env"]["alignRewardScale"],
        }

        self.dr_settings["noise"].update({
            # pos in meters, quat/dof in radians
            "cupA_rimpos": [0, 0.01], # in all directions
            "cupA_rimquat": [0, np.pi/60], # 3 degrees
            "cupA_to_cupB_pos": [0, 0.01], # in all directions
        })

        self.dr_settings["spatial"].update({
            "cupA_pos": [0., 0.],
            "cupA_quat": [0., 0.],
            "cupB_pos": [0., self.start_position_noise],
            "cupB_quat": [0., self.start_rotation_noise],
        })

        # Dimensions
        # obs include: cupA_pose (7) + cupA_to_cupB_pos (3) + eef_pose (7) + gripper_mode (1)
        self.cfg["env"]["numObservations"] = 18
        self.cfg["env"]["numObservations"] += 21 if not self.control_type == "osc" else 0
        self.obs_act_rew_init()

        super().__init__(config=self.cfg, rl_device=rl_device, sim_device=sim_device, graphics_device_id=graphics_device_id, headless=headless, virtual_screen_capture=virtual_screen_capture, force_render=force_render)
            
        # custom init
        self.timeaware_init()
        self.env_extra_init()
        self._warmup_env()
    
    
    def env_extra_init(self):
        self.franka_default_dof_pos = to_torch(
            [0.0200,  0.4040, -0.2257, -2.4885,  0.4908,  3.0764,  0.5566, 0.01, 0.01], device=self.device
        )

        # Computed by the simulation transformation
        self.default_eef2cupA_pos = to_torch([-0.0153, -0.0344, 0.0680], device=self.device).unsqueeze(0).repeat(self.num_envs, 1)
        self.default_eef2cupA_quat = to_torch([-0.9592,  0.2641, -0.0975,  0.0252], device=self.device).unsqueeze(0).repeat(self.num_envs, 1)

        self.focus_names = ["cupA", "cupB", "gm_single"] if not self.use_real_pure else ["cupA", "cupB"]
        self.focus_linvel_names = [f"{focus_name}_linvel_norm" for focus_name in self.focus_names]
        self.focus_linacc_names = [f"{focus_name}_linacc_norm" for focus_name in self.focus_names]
        # self.apply_force_handle = "cupA_handle"

        # Transformation buffer
        self.cupA_base2rim = torch.tensor([0., 0., self.cupA_h], device=self.device).unsqueeze(0).repeat(self.num_envs, 1)
        self.cupB_base2rim = torch.tensor([0., 0., self.cupB_h], device=self.device).unsqueeze(0).repeat(self.num_envs, 1)
        self.cupA_base2grasp_lf = torch.tensor(
            [0, 0, max(self.cupA_h-0.02, 0)], 
            device=self.device).unsqueeze(0).repeat(self.num_envs, 1) # Add the recommend grasp position
        self.cupA_base2grasp_rf = torch.tensor(
            [0, 0, max(self.cupA_h-0.02, 0)], 
            device=self.device).unsqueeze(0).repeat(self.num_envs, 1)
        self.cupA_base2grasp_mid = (self.cupA_base2grasp_lf + self.cupA_base2grasp_rf) / 2

    
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

        # Crate cup(s)
        cup_shape = "round"
        n_slices = 6
        # Warning: If change cup size the grasp pose will also need to adjusted. 
        thickness = 0.005
        cupA_mid2wallcenter = 0.035
        cupA_mid2wallouter = cupA_mid2wallcenter + thickness / 2
        cup_r = cupA_mid2wallouter / np.cos(np.pi / n_slices)
        self.cupA_r = self.cupB_r = float(cup_r)
        self.cupA_h = self.cupB_h = cup_h = 0.1
        self.cup_thickness = thickness
        mass = 0.05
        # Generate cup using primitive objects for approximation in the simulation
        cupA_asset_fpath = cupB_asset_fpath = create_hollow_cylinder(
            name="cup",
            size=[cupA_mid2wallcenter, cup_h],
            thickness=thickness,
            mass=mass,
            n_slices=n_slices,
            shape=cup_shape,
            use_lid=False,
            transparent_walls=False,
            generate_urdf=True,
            unique_urdf_name=False,
            asset_root_path=asset_root
        )

        # Generate cup using completed mesh for printing and sim2real
        create_hollow_cylinder_mesh(
            radius=cup_r,
            height=cup_h,
            thickness=thickness,
            n_sides=n_slices,
            save_path=os.path.join(asset_root, "urdf/procedural/real_cup.stl")
        )
        
        if self.cfg.get("use_container", False):
            cup_shape = "square"
            n_slices = 4
            cupB_mid2wallcenter = 0.05
            cupB_mid2wallouter = cupB_mid2wallcenter + thickness / 2
            cup_r = cupB_mid2wallouter / np.cos(np.pi / n_slices)
            self.cupB_r = float(cup_r)
            cupB_asset_fpath = create_hollow_cylinder(
                name="cupB",
                size=[cupB_mid2wallcenter, cup_h/2],
                thickness=thickness,
                mass=mass,
                n_slices=n_slices,
                shape=cup_shape,
                use_lid=False,
                transparent_walls=False,
                generate_urdf=True,
                unique_urdf_name=False,
                asset_root_path=asset_root
            )

        # Create cupA asset
        cup_opts = gymapi.AssetOptions()
        # cup_opts.vhacd_enabled = True # If using mesh urdf with holes, vhacd is required
        cup_opts.collapse_fixed_joints = True
        cup_opts.fix_base_link = False
        # Not sure about how much effect will apply on using mesh materials or not
        cup_opts.use_mesh_materials = True

        cup_assets = []
        for cup_asset_fpath in [cupA_asset_fpath, cupB_asset_fpath]:
            cup_asset = self.gym.load_asset(self.sim, asset_root, cup_asset_fpath, cup_opts)
            cup_props = self.gym.get_asset_rigid_shape_properties(cup_asset)[0]
            # Not sure about how much effect will apply on the contact_offset and rest_offset and why thickness matters
            cup_props.friction = 1.0
            cup_props.contact_offset = -1
            cup_props.thickness = 0.001
            self.gym.set_asset_rigid_shape_properties(cup_asset, [cup_props])
            cup_assets.append(cup_asset)
        cupA_asset, cupB_asset = cup_assets

        cupA_color = gymapi.Vec3(0.6, 0.1, 0.0)
        cupB_color = gymapi.Vec3(0.0, 0.4, 0.1)

        # Create ball asset
        self.num_gms = num_gms = self.cfg.get("num_gms", 1)
        self.gm_r = gm_r = 0.005
        gm_opts = gymapi.AssetOptions()
        gm_opts.density = 2000
        gm_asset = self.gym.create_sphere(self.sim, gm_r, gm_opts)
        # Watch out the default gm_prop.contact_offset is -1! While the new prop_group is 0
        gm_prop = self.gym.get_asset_rigid_shape_properties(gm_asset)[0]
        gm_prop.restitution = 0.02
        # Not sure about the contact_offset and rest_offset affects or not
        gm_prop.thickness = 0.001
        self.gym.set_asset_rigid_shape_properties(gm_asset, [gm_prop])

        gm_color = gymapi.Vec3(0.0, 0.7, 0.9)

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
        cupA_start_pose = gymapi.Transform()
        cupA_start_pose.p = gymapi.Vec3(-1.0, 0.0, 0.0)
        cupA_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
        cupB_start_pose = gymapi.Transform()
        cupB_start_pose.p = gymapi.Vec3(1.0, 0.0, 0.0)
        cupB_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        # Define start pose for gms (doesn't really matter since it gets overridden during reset() anyways)
        gm_start_pose = gymapi.Transform()
        gm_start_pose.p = gymapi.Vec3(-1.0, 0.0, 0.0)
        gm_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        # compute aggregate size
        num_franka_bodies = self.gym.get_asset_rigid_body_count(franka_asset)
        num_franka_shapes = self.gym.get_asset_rigid_shape_count(franka_asset)
        num_cup_bodies = self.gym.get_asset_rigid_body_count(cup_asset)
        num_cup_shapes = self.gym.get_asset_rigid_shape_count(cup_asset)
        num_gm_bodies = self.gym.get_asset_rigid_body_count(gm_asset)
        num_gm_shapes = self.gym.get_asset_rigid_shape_count(gm_asset)
        max_agg_bodies = num_franka_bodies + 2 * num_cup_bodies + num_gms * num_gm_bodies + 3     # 2 for table, table stand, ws_stand
        max_agg_shapes = num_franka_shapes + 2 * num_cup_shapes + num_gms * num_gm_shapes + 3     # 2 for table, table stand, ws_stand

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

            # Create cups; ID will not change in the environment
            self._cupA_id = self.gym.create_actor(env_ptr, cupA_asset, cupA_start_pose, "cupA", i, 0, 0)
            self._cupB_id = self.gym.create_actor(env_ptr, cupB_asset, cupB_start_pose, "cupB", i, 0, 0)
            # Set colors
            for b_idx in range(num_cup_bodies):
                self.gym.set_rigid_body_color(env_ptr, self._cupA_id, b_idx, gymapi.MESH_VISUAL, cupA_color)
                self.gym.set_rigid_body_color(env_ptr, self._cupB_id, b_idx, gymapi.MESH_VISUAL, cupB_color)
            # Create granular medias
            self._gm_ids = []
            for gm_idx in range(num_gms):
                gm_id = self.gym.create_actor(env_ptr, gm_asset, gm_start_pose, f"gm_{gm_idx}", i, 0, 0)
                self._gm_ids.append(gm_id)
                # Set color
                self.gym.set_rigid_body_color(env_ptr, gm_id, 0, gymapi.MESH_VISUAL, gm_color)

            if self.aggregate_mode > 0:
                self.gym.end_aggregate(env_ptr)

            # Store the created env pointers
            self.envs.append(env_ptr)
            self.frankas.append(franka_actor)

        # Setup init state buffer
        self._init_cupA_state = torch.zeros(self.num_envs, 13, device=self.device)
        self._init_cupB_state = torch.zeros(self.num_envs, 13, device=self.device)
        self._init_gms_state = torch.zeros(self.num_envs, num_gms, 13, device=self.device)
        self.fk_init_dof = torch.zeros((self.num_envs, 9), dtype=torch.float, device=self.device)

        # Setup data
        self.init_data()


    def init_data(self):
        # Setup sim handles
        env_ptr = self.envs[0]
        franka_handle = 0
        self.common_data_init(env_ptr, franka_handle)
        self.link_handles.update({
            # Cups
            "cupA_handle": self.gym.find_actor_rigid_body_handle(env_ptr, self._cupA_id, "cup_base"),
            "cupB_handle": self.gym.find_actor_rigid_body_handle(env_ptr, self._cupB_id, "cup_base"),
        })
       
        # Robot base transformation stored for later use
        self._franka_state = self._root_state[:, franka_handle, :]
        self._cupA_state = self._root_state[:, self._cupA_id, :]
        self._cupB_state = self._root_state[:, self._cupB_id, :]
        # Watch out the advanced indexing or slicing. A list of ids will return a copy tensor!
        self._gm_states = self._root_state[:, self._gm_ids[0]:self._gm_ids[-1]+1, :]
        # Initialize states
        cupA_size = to_torch([self.cupA_r*2, self.cupA_h, self.cup_thickness], device=self.device).expand(self.num_envs, 3)
        cupB_size = to_torch([self.cupB_r*2, self.cupB_h, self.cup_thickness], device=self.device).expand(self.num_envs, 3)
        self.states.update({
            "cupA_size": cupA_size,
            "cupB_size": cupB_size,
        })

    
    def _update_task_prev_states(self, real=False):
        states, prev_states = self.get_states_dict(real=real)
        for name in ["cupA_pos", "cupA_quat", "cupB_pos", "cupB_quat", "gm_single_pos", "cupA_linvel", "cupB_linvel", "gm_single_linvel"]:
            if name not in states: continue
            prev_states[name] = mix_clone(states[name])
    

    def _update_task_states(self, reset_ids=[]):
        # Compute the distance between the rim of cupA and cupB and other added transformations
        cupA_pos, cupA_quat = self._cupA_state[:, :3], self._cupA_state[:, 3:7]
        cupB_pos, cupB_quat = self._cupB_state[:, :3], self._cupB_state[:, 3:7]
        cupA_rimquat, cupA_rimpos = tf_combine(cupA_quat, cupA_pos, self.unit_quat, self.cupA_base2rim)
        cupB_rimquat, cupB_rimpos = tf_combine(cupB_quat, cupB_pos, self.unit_quat, self.cupB_base2rim)
        cupA_graspmid_pos = tf_apply(cupA_quat, cupA_pos, self.cupA_base2grasp_mid)
        cupA_grasplf_pos = tf_apply(cupA_quat, cupA_pos, self.cupA_base2grasp_lf)
        cupA_grasprf_pos = tf_apply(cupA_quat, cupA_pos, self.cupA_base2grasp_rf)

        r2cupA_quat, r2cupA_pos = self.point2frankaBase(cupA_quat, cupA_pos)
        r2cupB_quat, r2cupB_pos = self.point2frankaBase(cupB_quat, cupB_pos)
        r2cupA_rimquat, r2cupA_rimpos = self.point2frankaBase(cupA_rimquat, cupA_rimpos)
        _, r2cupA_graspmid_pos = self.point2frankaBase(self.unit_quat, cupA_graspmid_pos)
        r2cupA_grasplf_quat, r2cupA_grasplf_pos = self.point2frankaBase(self.unit_quat, cupA_grasplf_pos)
        r2cupA_grasprf_quat, r2cupA_grasprf_pos = self.point2frankaBase(self.unit_quat, cupA_grasprf_pos)
        r2cupB_rimquat, r2cupB_rimpos = self.point2frankaBase(cupB_rimquat, cupB_rimpos)

        r2gm_pos = self.points2frankaBase(self._gm_states[:, :, :3])

        self.states.update({
            # Cubes
            "cupA_quat": r2cupA_quat,
            "cupA_pos": r2cupA_pos,
            "cupA_rimpos": r2cupA_rimpos,
            "cupA_rimquat": r2cupA_rimquat,
            "cupA_graspmid_pos": r2cupA_graspmid_pos,
            "cupA_grasplf_pos": r2cupA_grasplf_pos,
            "cupA_grasplf_quat": r2cupA_grasplf_quat,
            "cupA_grasprf_pos": r2cupA_grasprf_pos,
            "cupA_grasprf_quat": r2cupA_grasprf_quat,
            
            "cupB_quat": r2cupB_quat,
            "cupB_pos": r2cupB_pos,
            "cupB_rimpos": r2cupB_rimpos,
            "cupB_rimquat": r2cupB_rimquat,
            "cupA_to_cupB_pos": r2cupB_rimpos - r2cupA_rimpos,
            "gm_pos": r2gm_pos,
            "gm_single_pos": r2gm_pos[:, 0, :],
        })

        # ----------------------------------------------------
        # Update states in the world frame for rendering later
        self.world_states.update({
            # Cubes
            "cupA_quat": self._cupA_state[:, 3:7],
            "cupA_pos": self._cupA_state[:, :3],
            "cupA_rimpos": cupA_rimpos,
            "cupA_graspmid_pos": cupA_graspmid_pos,
            "cupA_grasplf_pos": cupA_grasplf_pos,
            "cupA_grasprf_pos": cupA_grasprf_pos,
            "cupA_linvel": self._cupA_state[:, 7:10],
            "cupB_quat": self._cupB_state[:, 3:7],
            "cupB_pos": self._cupB_state[:, :3],
            "cupB_rimpos": cupB_rimpos,
            "cupB_linvel": self._cupB_state[:, 7:10],
            "gm_pos": self._gm_states[:, :, :3],
        })


    def _update_diff_states(self, real=False):
        # TODO: convert to the rim velocity
        # r2cupA_linvel = self.vec2frankaBase(self._cupA_state[:, 7:10])
        # r2cupB_linvel = self.vec2frankaBase(self._cupB_state[:, 7:10])

        states, prev_states = self.get_states_dict(real=real)
        # We use position to compute the linear velocity instead of directly using the linear velocity in isaacgym
        r2cupA_pos = states["cupA_pos"]
        r2cupB_pos = states["cupB_pos"]
        prev_r2cupA_pos = prev_states["cupA_pos"] if "cupA_pos" in prev_states else r2cupA_pos
        prev_r2cupB_pos = prev_states["cupB_pos"] if "cupB_pos" in prev_states else r2cupB_pos
        r2cupA_linvel = (r2cupA_pos - prev_r2cupA_pos) / self.ctrl_dt
        r2cupB_linvel = (r2cupB_pos - prev_r2cupB_pos) / self.ctrl_dt
        
        # Use ema filter to smooth the noisy linear velocity for real world. 
        # Simulation smoothness is in collecting_observation after applying position noise.
        if real:
            prev_r2cupA_linvel = states["cupA_linvel"] if "cupA_linvel" in states else r2cupA_linvel
            prev_r2cupB_linvel = states["cupB_linvel"] if "cupB_linvel" in states else r2cupB_linvel
            r2cupA_linvel = ema_filter(r2cupA_linvel, prev_r2cupA_linvel, alpha=0.9)
            r2cupB_linvel = ema_filter(r2cupB_linvel, prev_r2cupB_linvel, alpha=0.9)
        else:
            # For simulation, we also compute the linear velocity of the gm in cupA for behavior optimization
            r2gm_single_pos = states["gm_single_pos"]
            prev_r2gm_single_pos = prev_states["gm_single_pos"] if "gm_single_pos" in prev_states else r2gm_single_pos
            r2gm_single_linvel = (r2gm_single_pos - prev_r2gm_single_pos) / self.ctrl_dt
            first_gm_in_cupA = is_in_cup(states["gm_pos"], states["cupA_pos"], states["cupA_rimpos"], states["cupA_size"])[:, 0]
            r2gm_single_linvel[~first_gm_in_cupA] = 0.0 # If the gm is in out of cupA, we wont detect its velocity
            states.update({
                "gm_single_linvel": r2gm_single_linvel,
                "gm_single_linvel_norm": mix_norm(r2gm_single_linvel, dim=-1, keepdim=True),
            })
        
        states.update({
            "cupA_linvel": r2cupA_linvel,
            "cupB_linvel": r2cupB_linvel,
            "cupA_linvel_norm": mix_norm(r2cupA_linvel, dim=-1, keepdim=True),
            "cupB_linvel_norm": mix_norm(r2cupB_linvel, dim=-1, keepdim=True),
        })

        # Update previous states to compute acceleration
        if "cupA_linvel" in states and "cupA_linvel" in prev_states:
            r2cupA_linacc = (states["cupA_linvel"] - prev_states["cupA_linvel"]) / self.ctrl_dt
            r2cupB_linacc = (states["cupB_linvel"] - prev_states["cupB_linvel"]) / self.ctrl_dt
            states.update({
                "cupA_linacc_norm": mix_norm(r2cupA_linacc, dim=-1, keepdim=True),
                "cupB_linacc_norm": mix_norm(r2cupB_linacc, dim=-1, keepdim=True),
            })


    def compute_task_reward(self):
        """
        Watch out! All states info are in the robot base frame!
        """
        cfg = self.cfg
        states = self.states
        prev_states = self.prev_states
        reward_settings = self.reward_settings
        
        # Compute per-env physical parameters
        cupA_r = states["cupA_size"][:, 0] / 2
        cupA_h = states["cupA_size"][:, 1]
        cupB_r = states["cupB_size"][:, 0] / 2
        cupB_h = states["cupB_size"][:, 1]
        init_height = 0.
        target_height = cupB_h * 1.5

        # Scene stability
        cupA_linvel_norm = states["cupA_linvel_norm"].flatten()
        cupA_linacc_norm = states["cupA_linacc_norm"].flatten()
        cupB_linvel_norm = states["cupB_linvel_norm"].flatten()
        cupB_linacc_norm = states["cupB_linacc_norm"].flatten()
        table_contact_forces_norm = torch.norm(states["table_contact_forces"], dim=-1)

        # Why we need to design reward for grasping:
        # We want to provide guidance and avoid some the reward hacking behavior. The grasp is serving for the whole task success instead of just "grasp". 
        # We can not constrain the exact grasp pose because it might do damage to the later execution. 
        # Instead, we are doing rl to let the policy decide which grasp pose is the best one for the task.

        # We compute the minimum distance between the gripper and the cupA as long as one is inside and the other one is outside
        d = torch.norm(self.states["cupA_graspmid_pos"] - self.states["eef_pos"], dim=-1)
        d_lf = torch.norm(states["cupA_graspmid_pos"] - states["eef_lf_pos"], dim=-1)
        d_rf = torch.norm(states["cupA_graspmid_pos"] - states["eef_rf_pos"], dim=-1)
        is_gripepr_approaching = d <= 0.02
        is_gripper_closing = self._gripper_mode==1.
        dist_reward = 0.5 * (1 - torch.tanh(10.0 * d) + \
                             0.5 * is_gripper_closing * is_gripepr_approaching + \
                             0.5 * ~is_gripper_closing * ~is_gripepr_approaching)
        
        # Alignment reward bonus (Watchout normalization!!): 
        # direction_v = normalize(states["cupA_grasplf_pos"] - states["cupA_grasprf_pos"])
        # gripper_v = normalize(states["eef_rf_pos"] - states["eef_lf_pos"])
        # # # The angle between the gripper and the cupA should be close to 90. The dot product should be close to 0
        # ori_dist = batch_dot(direction_v, gripper_v).abs()
        # ori_reward = 0.2 * ori_dist
        # dist_reward = (dist_reward + ori_reward) / 1.2

        # reward for lifting cupA; Remember that the frame of cupA is already at the bottom!
        # All height is relative to the table surface
        cupA_bot_height = states["cupA_pos"][:, 2] - self.franka2ws_surface_height
        cupA_rim_height = states["cupA_rimpos"][:, 2] - self.franka2ws_surface_height
        cupA_height = torch.where(cupA_bot_height <= cupA_rim_height, cupA_bot_height, cupA_rim_height) # The minimum height of the cupA
        lift_reward = torch.clamp(cupA_height-init_height, 
                                  min=torch.zeros_like(cupA_height), 
                                  max=target_height-init_height)
        cupA_lifted = cupA_height >= target_height

        # how closely aligned cupA is to cupB the port distance (only provided if cupA is lifted)
        d_ab = torch.norm(states["cupA_to_cupB_pos"], dim=-1)
        states["approaching_dist"] = d_ab_xy = torch.norm(states["cupA_to_cupB_pos"][:, :2], dim=-1)
        cupA_zaxis = normalize(states["cupA_rimpos"] - states["cupA_pos"])
        cupB_zaxis = normalize(states["cupB_rimpos"] - states["cupB_pos"])
        ori_dist = torch.abs(batch_dot(cupA_zaxis, cupB_zaxis) + 0.5) # minimize ori_dist to be -0.5 (120 degree)
        align_reward_pos = 1 - torch.tanh(10.0 * d_ab)
        align_reward_rot = (1 - torch.tanh(2.0 * ori_dist)) * (d_ab_xy < (cupB_r+cupA_r))
        align_reward = (align_reward_pos + align_reward_rot) / 2 * cupA_lifted

        # Potential based reward trial
        if self.cfg.get("use_potential_r", False):
            cur_dist_reward, cur_lift_reward, cur_align_reward = dist_reward.clone(), lift_reward.clone(), align_reward.clone()
            if "prev_dist_reward" in self.prev_states:
                dist_reward = cur_dist_reward - self.prev_states["prev_dist_reward"]
                lift_reward = cur_lift_reward - self.prev_states["prev_lift_reward"]
                align_reward = cur_align_reward - self.prev_states["prev_align_reward"]
            self.prev_states.update({
                "prev_dist_reward": cur_dist_reward,
                "prev_lift_reward": cur_lift_reward,
                "prev_align_reward": cur_align_reward,
            })

        gms_poured = ~is_in_cup(states["gm_pos"], states["cupA_pos"], states["cupA_rimpos"], states["cupA_size"])
        gms_in_cupB = is_in_cup(states["gm_pos"], states["cupB_pos"], states["cupB_rimpos"], states["cupB_size"])
        gms_on_table = ~gms_in_cupB & ((states["gm_pos"][:, :, 2] - self.franka2ws_surface_height) < self.gm_r * 4)
        gms_succecced = gms_poured & gms_in_cupB
        gms_missed = gms_poured & gms_on_table
        num_gms_succeded = gms_succecced.sum(dim=-1)
        all_gms_poured = gms_poured.all(dim=-1)
        all_gms_succecced = gms_succecced.all(dim=-1)
        all_gms_missed = gms_missed.all(dim=-1)

        cupA_is_in_hand = d <= (cupA_r * 2)
        cupA_is_stable = cupA_linvel_norm < 0.1
        cupA_is_stand = cupA_rim_height > cupA_r * 1.5 # Keep the cupA in the upright position
        cupB_is_stable = cupB_linvel_norm < 0.02
        cupB_is_stand = (states["cupB_pos"][:, 2] - self.franka2ws_surface_height) <= cupB_r * np.sin(np.pi/6) # Keep the cupB in the upright position
        cupA_is_close_to_cupB = d_ab <= 0.1
        cupA_hits_cupB = ~cupB_is_stable & (num_gms_succeded==0)
        
        single_check = all_gms_poured & cupB_is_stable & cupA_is_close_to_cupB
        holding_reward = single_check * num_gms_succeded

        # Check the success conditions

        self.continuous_check_buf[single_check] += 1
        not_continuous_success = (self.continuous_check_buf > 0) & ~single_check
        ending_conditions = self.continuous_check_buf >= cfg.get("check_steps", int(2/self.ctrl_dt))
        success_conditions = all_gms_succecced & ending_conditions

        # regularization on the actions (summed for each environment)
        action_penalty = torch.norm(self.actions, dim=-1)
        force_penalty = torch.zeros_like(table_contact_forces_norm)
        if reward_settings["r_force_penalty_scale"] > 0: # reduce computation if not needed
            force_penalty = torch.where(table_contact_forces_norm > self.MAX_CONTACT_FORCE_NORM, table_contact_forces_norm, force_penalty)

        # Failed conditions; We do not add cupB_is_stable since we allow the cupB to move a little bit during manipulation
        violated_conditions = all_gms_missed | ~cupA_is_in_hand | ~cupB_is_stand | ~cupA_is_stand | cupA_hits_cupB | not_continuous_success | \
                              ~is_under_valid_vel(cupA_linvel_norm, self.MAX_VEL_NORM) | ~is_under_valid_vel(cupB_linvel_norm, self.MAX_VEL_NORM) | (self.progress_buf >= self.max_episode_length - 1)
        if self.cur_curri_ratio==1.:
            violated_conditions |= ~is_under_valid_contact(table_contact_forces_norm, self.MAX_CONTACT_FORCE_NORM)
        
        if self.cfg.get("constrain_grasp", False):
            gripper_in_cupA = is_in_cup(states["eef_lf_pos"], states["cupA_pos"], states["cupA_rimpos"], states["cupA_size"]).squeeze(-1) | \
                              is_in_cup(states["eef_rf_pos"], states["cupA_pos"], states["cupA_rimpos"], states["cupA_size"]).squeeze(-1)
            violated_conditions = violated_conditions | gripper_in_cupA

        # Compute resets
        reset_conditions = ending_conditions | violated_conditions

        # Compose rewards
        # We either provide the stack reward or the align + dist reward
        rewards = torch.where(
            reset_conditions,
            
            reward_settings["r_success"] * success_conditions,
            
            reward_settings["r_dist_scale"] * dist_reward + reward_settings["r_lift_scale"] * lift_reward + \
            reward_settings["r_align_scale"] * align_reward + reward_settings["r_hold_scale"] * holding_reward - \
            reward_settings["r_action_penalty_scale"] * action_penalty - reward_settings["r_force_penalty_scale"] * force_penalty,
        )
        # Compute buffer
        reset_buf = torch.where(reset_conditions > 0, torch.ones_like(self.reset_buf), self.reset_buf)
        success_buf = torch.where(success_conditions > 0, torch.ones_like(self.reset_buf), torch.zeros_like(self.reset_buf))
        return rewards, reset_buf, success_buf
        

    def get_taskobs_names(self):
        obs_names = ["cupA_rimpos", "cupA_rimquat", "cupA_to_cupB_pos", "eef_pos", "eef_quat"]
        obs_names += [] if self.control_type == "osc" else ["q", "prev_tgtq", "prev_dq"]
        obs_names += ["gripper_mode"]
        return obs_names
    

    def task_prim_obs_init(self):
        self.cfg["env"]["numStates"] += 3 # gm_single_pos (3)
    
    
    def add_priv_taskobs(self, obs_names):
        obs_names += ["gm_single_pos"]
        return obs_names
    

    def reset_idx(self, env_ids):
        config_index = self.get_config_idx(env_ids)
        new_cfg_env_ids, config_index = self.filter_env_ids(env_ids, config_index)

        # Compute the initial states of all movable objects
        if len(new_cfg_env_ids) > 0:
            # Reset environment configurations and time related states that require speed ratio
            self._reset_timeaware_states(new_cfg_env_ids, config_index)
            if config_index is not None:
                self._init_cupA_state[new_cfg_env_ids] = self._apply_cube_state_noise(self.env_configs["cupA_state"][config_index].clone(), new_cfg_env_ids) \
                                                           if (self.training and self.cfg.get("add_cube_noise", False)) else self.env_configs["cupA_state"][config_index].clone()
                self._init_cupB_state[new_cfg_env_ids] = self._apply_cube_state_noise(self.env_configs["cupB_state"][config_index].clone(), new_cfg_env_ids) \
                                                           if (self.training and self.cfg.get("add_cube_noise", False)) else self.env_configs["cupB_state"][config_index].clone()
                init_fk_pos = self.env_configs["franka_state"][config_index].clone()
                self._init_gms_state[new_cfg_env_ids] = self._sample_gms_state(new_cfg_env_ids) # We do not change the init gms state
            else:
                # Update brand new cube states (self._init_cube_state is inside the function)
                self._init_cupA_state[new_cfg_env_ids] = self._reset_init_cube_state("cupA", 
                                                                                     self.states["cupA_size"],
                                                                                     torch.zeros_like(self.states["cupA_size"][:, 2]),
                                                                                     self._init_cupB_state, 
                                                                                     self.states["cupB_size"] if not self.cfg.get("use_container", False) else self.states["cupA_size"], 
                                                                                     env_ids=new_cfg_env_ids, 
                                                                                     check_valid=False)
                self._init_cupB_state[new_cfg_env_ids] = self._reset_init_cube_state("cupB", 
                                                                                     self.states["cupB_size"], 
                                                                                     torch.zeros_like(self.states["cupB_size"][:, 2]), 
                                                                                     self._init_cupA_state, 
                                                                                     self.states["cupA_size"], 
                                                                                     env_ids=new_cfg_env_ids, 
                                                                                     check_valid=True)

                self._init_gms_state[new_cfg_env_ids] = self._sample_gms_state(new_cfg_env_ids)

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
        self._cupA_state[env_ids] = self._init_cupA_state[env_ids]
        self._cupB_state[env_ids] = self._init_cupB_state[env_ids]
        self._gm_states[env_ids] = self._init_gms_state[env_ids]
        # Reset the internal obs accordingly
        self._q[env_ids, :] = self.fk_init_dof[env_ids, :]
        self._qd[env_ids, :] = torch.zeros_like(self._qd[env_ids])

        # Set any position control to the current position, and any vel / effort control to be 0
        # NOTE: Task takes care of actually propagating these controls in sim using the SimActions API
        self._pos_control[env_ids, :] = self.fk_init_dof[env_ids, :]
        self._effort_control[env_ids, :] = torch.zeros_like(self.fk_init_dof[env_ids, :])

        # Deploy arm updates; arm id is always 0
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

        # Deploy movable updates; all movable objects are loaded since the cupA
        multi_env_ids_cups_int32 = self._global_indices[env_ids, self._cupA_id:].flatten()
        self.gym.set_actor_root_state_tensor_indexed(self.sim, 
                                                     gymtorch.unwrap_tensor(self._root_state),
                                                     gymtorch.unwrap_tensor(multi_env_ids_cups_int32), 
                                                     len(multi_env_ids_cups_int32))

        # Reset buffers (in this env, we keep the gripper closed)
        self._reset_bufs(env_ids)
        self._gripper_mode[env_ids] = 1
        self._gripper_mode_temp[env_ids] = 1

        # Record the initial configurations
        if self.is_ready_to_record(new_cfg_env_ids):
            self.record_init_configs(new_cfg_env_ids)

        # Need one step to refresh the dof states; Otherwise, the states (such as eef pose) will be outdated
        self.gym.simulate(self.sim)
        if self.device == 'cpu':
            self.gym.fetch_results(self.sim, True)
        if self.force_render:
            self.render()


    def reset_real_cup_states(self, cupA_pose, cupB_pose):
        # Reset the cube states to the given pose
        sampled_cupA_state = torch.zeros_like(self._cupA_state)
        sampled_cupB_state = torch.zeros_like(self._cupB_state)
        sampled_cupA_state[:, :7] = cupA_pose[:, :7]
        sampled_cupB_state[:, :7] = cupB_pose[:, :7]  
        self._cupA_state[:] = sampled_cupA_state
        self._cupB_state[:] = sampled_cupB_state

        # Reset the sim states
        cupA_ids = self._global_indices[:, self._cupA_id]
        cupB_ids = self._global_indices[:, self._cupB_id]
        multi_obj_ids = torch.cat([cupA_ids, cupB_ids], dim=0)
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
        init_configs["cupA_state"] = []
        init_configs["cupB_state"] = []
        init_configs["franka_state"] = []
    

    def _record_task_init_configs(self, env_ids):
        init_configs = self.extras["init_configs"]
        init_configs["cupA_state"].extend(self._init_cupA_state[env_ids].cpu().tolist())
        init_configs["cupB_state"].extend(self._init_cupB_state[env_ids].cpu().tolist())
        init_configs["franka_state"].extend(self._q[env_ids].cpu().tolist())
    

    def pre_physics_step(self, actions):
        self._gripper_mode[:] = 1
        self._gripper_mode_temp[:] = 1
        actions[:, -1] = 1. # Keep the gripper closing
        super().pre_physics_step(actions)


    def debug_viz(self):
        self.gym.clear_lines(self.viewer)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        # Grab relevant states to visualize
        eef_pos = self.states["eef_pos"]
        eef_rot = self.states["eef_quat"]
        cupA_pos = self.states["cupA_pos"]
        cupA_rot = self.states["cupA_quat"]
        cupB_pos = self.states["cupB_pos"]
        cupB_rot = self.states["cupB_quat"]

        cupA_grasplf_pos = self.states["cupA_grasplf_pos"]
        cupA_grasplf_quat = self.states["cupA_grasplf_quat"]
        cupA_grasprf_pos = self.states["cupA_grasprf_pos"]
        cupA_grasprf_quat = self.states["cupA_grasprf_quat"]

        # Plot visualizations
        for i in range(self.num_envs):
            for pos, rot in zip((eef_pos, cupA_pos, cupB_pos), (eef_rot, cupA_rot, cupB_rot)):
                px = (pos[i] + quat_apply(rot[i], to_torch([1, 0, 0], device=self.device) * 0.2)).cpu().numpy()
                py = (pos[i] + quat_apply(rot[i], to_torch([0, 1, 0], device=self.device) * 0.2)).cpu().numpy()
                pz = (pos[i] + quat_apply(rot[i], to_torch([0, 0, 1], device=self.device) * 0.2)).cpu().numpy()

                p0 = pos[i].cpu().numpy()
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], px[0], px[1], px[2]], [0.85, 0.1, 0.1])
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], py[0], py[1], py[2]], [0.1, 0.85, 0.1])
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], pz[0], pz[1], pz[2]], [0.1, 0.1, 0.85])
        
        _, w2cupA_grasplf_pos = tf_combine(self._franka_state[:, 3:7], self._franka_state[:, :3], cupA_grasplf_quat, cupA_grasplf_pos)
        _, w2cupA_grasprf_pos = tf_combine(self._franka_state[:, 3:7], self._franka_state[:, :3], cupA_grasprf_quat, cupA_grasprf_pos)
        self.draw_point(w2cupA_grasplf_pos[0])
        self.draw_point(w2cupA_grasprf_pos[0])

    
    #####################################################################
    ##=========================utils functions=========================##
    #####################################################################
    def _sample_gms_state(self, env_ids):
        gms_state = torch.zeros(len(env_ids), self.num_gms, 13, device=self.device)
        # Remeber to set orientation to 1.0 (it will not show error but will make the object disappear)
        gms_state[:, :, 6] = 1.0
        # # Randomly initialize the gms state into the cupA (no collision)
        cupA_state = self._init_cupA_state[env_ids]
        cupA_bottom2gms = self.initialize_spheres_stable_torch(self.num_gms, self.gm_r, self.cupA_r, self.cupA_h, self.cup_thickness)
        # Initialize the gms state from (num_envs, 13) to (num_envs, num_gms, 13)
        world2gms = tf_apply(cupA_state[:, 3:7].unsqueeze(1).repeat(1, self.num_gms, 1), 
                             cupA_state[:, :3].unsqueeze(1).repeat(1, self.num_gms, 1), 
                             cupA_bottom2gms.repeat(len(env_ids), 1, 1))
        gms_state[:, :, :3] = world2gms
        return gms_state
    
    def initialize_spheres_random_torch(
        self,
        num_spheres, 
        sphere_radius, 
        cylinder_radius, 
        cylinder_height, 
        cylinder_thickness, 
        max_attempts=10000,
        device="cuda"
    ):
        """
        Rejection sampling for random initial positions for spheres within a cylindrical cup using PyTorch.

        Args:
            num_spheres (int): Number of spheres to initialize.
            sphere_radius (float): Radius of each sphere.
            cylinder_radius (float): Inner radius of the cylinder (excluding thickness).
            cylinder_height (float): Height of the cylinder cup.
            cylinder_thickness (float): Thickness of the cylinder wall.
            max_attempts (int): Maximum number of attempts to find valid positions.
            device (str): Device to run the computations on ('cpu' or 'cuda').

        Returns:
            torch.Tensor: A tensor of shape (num_spheres, 3) containing (x, y, z) positions.
        """
        # Effective inner radius (accounting for thickness and sphere radius)
        effective_radius = cylinder_radius - cylinder_thickness - sphere_radius
        effective_height = cylinder_height - cylinder_thickness
        if effective_radius <= 0:
            raise ValueError(f"Effective cylinder radius must be greater than zero.")

        # Initialize positions tensor (x, y, z)
        positions = torch.empty((0, 3), device=device)

        for attempt in range(max_attempts):
            # Generate random positions in cylindrical coordinates
            r = torch.sqrt(torch.rand((num_spheres,), device=device)) * effective_radius  # Random radius
            theta = torch.rand((num_spheres,), device=device) * 2 * torch.pi  # Random angle
            # Give a large range for z to avoid the sphere being too close to the bottom and seems good for gm initialization
            z = 5 * torch.rand((num_spheres,), device=device) * effective_height  # Sample within the effective height
            
            # Convert cylindrical coordinates to Cartesian (x, y, z)
            x = r * torch.cos(theta)
            y = r * torch.sin(theta)
            new_positions = torch.stack((x, y, z), dim=1)  # Shape: (num_spheres, 3)
            
            # Check for collisions with existing spheres
            if positions.size(0) > 0:
                dist = torch.cdist(new_positions, positions)  # Pairwise distances between new and existing spheres
                min_dist = torch.min(dist, dim=1).values  # Minimum distance to any existing sphere
                valid_mask = min_dist >= 2 * sphere_radius  # Check if distances are valid
            else:
                valid_mask = torch.ones((num_spheres,), device=device, dtype=torch.bool)  # First batch is always valid

            # Check if all new positions are within valid boundaries
            boundary_mask = (r <= effective_radius) & (z >= sphere_radius) & (z <= effective_height - sphere_radius)
            valid_mask &= boundary_mask  # Combine boundary and collision masks

            # Append valid positions to the positions tensor
            valid_positions = new_positions[valid_mask]
            positions = torch.cat((positions, valid_positions), dim=0)

            # Stop if we have enough spheres
            if positions.size(0) >= num_spheres:
                positions[:, 2] += cylinder_thickness  # Shift z-coordinates to account for the cylinder base
                return positions[:num_spheres]  # Return only the required number of positions

        # If we exit the loop without finding enough valid positions, raise an error
        raise RuntimeError(f"Could not place all {num_spheres} spheres after {max_attempts} attempts.")
    

    def initialize_spheres_stable_torch(
        self,
        num_spheres, 
        sphere_radius, 
        cylinder_radius, 
        cylinder_height, 
        cylinder_thickness, 
        packing_efficiency=0.8,
        device="cuda"
    ):
        """
        Initialize spheres in stable layers within a cylindrical cup using PyTorch. We use hexagonal packing for efficiency.
        Spheres are placed from bottom to top in a stable configuration.

        Args:
            num_spheres (int): Number of spheres to initialize.
            sphere_radius (float): Radius of each sphere.
            cylinder_radius (float): Inner radius of the cylinder (excluding thickness).
            cylinder_height (float): Height of the cylinder cup.
            cylinder_thickness (float): Thickness of the cylinder wall.
            packing_efficiency (float): Efficiency factor for hexagonal packing (0.6-0.9).
            device (str): Device to run the computations on ('cpu' or 'cuda').

        Returns:
            torch.Tensor: A tensor of shape (num_spheres, 3) containing (x, y, z) positions.
        """
        # Effective inner radius (accounting for thickness and sphere radius)
        effective_radius = cylinder_radius - cylinder_thickness - sphere_radius
        effective_height = cylinder_height - cylinder_thickness
        
        if effective_radius <= 0:
            raise ValueError(f"Effective cylinder radius must be greater than zero.")
        
        # Calculate sphere spacing for hexagonal packing
        sphere_spacing = 2 * sphere_radius * packing_efficiency
        
        # Calculate how many spheres can fit in one layer (hexagonal packing)
        max_spheres_per_layer = self._calculate_max_spheres_per_layer(effective_radius, sphere_spacing)
        
        if max_spheres_per_layer == 0:
            raise ValueError("Cylinder too small to fit even one sphere per layer.")
        
        positions = torch.empty((0, 3), device=device)
        current_layer = 0
        spheres_placed = 0
        
        while spheres_placed < num_spheres:
            # Calculate z-position for current layer
            if current_layer == 0:
                # First layer: spheres rest on the bottom
                z_pos = sphere_radius + cylinder_thickness
            else:
                # Subsequent layers: calculate based on sphere stacking
                z_pos = sphere_radius + cylinder_thickness + current_layer * sphere_spacing * np.sqrt(3)/2  # 0.866 ≈ sqrt(3)/2
            
            # Check if we can fit another layer
            if z_pos + sphere_radius > cylinder_height:
                break
            
            # Generate positions for current layer
            spheres_needed = min(num_spheres - spheres_placed, max_spheres_per_layer)
            layer_positions = self._generate_layer_positions(
                spheres_needed, 
                effective_radius, 
                sphere_spacing, 
                z_pos, 
                current_layer,
                device
            )
            
            positions = torch.cat((positions, layer_positions), dim=0)
            spheres_placed += layer_positions.size(0)
            current_layer += 1
        
        if spheres_placed < num_spheres:
            print(f"Warning: Could only place {spheres_placed} out of {num_spheres} spheres due to space constraints.")
        
        return positions[:num_spheres] if spheres_placed >= num_spheres else positions

    def _calculate_max_spheres_per_layer(self, effective_radius, sphere_spacing):
        """Calculate maximum number of spheres that can fit in one layer using hexagonal packing."""
        if effective_radius < sphere_spacing / 2:
            return 0
        
        # Estimate using area-based calculation
        circle_area = np.pi * effective_radius**2
        sphere_area = np.pi * (sphere_spacing / 2)**2
        hexagonal_efficiency = 0.9069  # Theoretical maximum for hexagonal packing
        
        return max(1, int(circle_area / sphere_area * hexagonal_efficiency))

    def _generate_layer_positions(self, num_spheres, effective_radius, sphere_spacing, z_pos, layer_index, device):
        """Generate positions for spheres in a single layer using hexagonal packing pattern."""
        positions = torch.empty((0, 3), device=device)
        
        if num_spheres <= 0:
            return positions
        
        # Hexagonal packing pattern
        hex_radius = sphere_spacing / 2
        
        # Calculate rings of hexagonal packing
        spheres_placed = 0
        ring = 0
        
        # Offset alternating layers for better stacking
        layer_rotation = (layer_index % 2) * (np.pi / 6)
        
        while spheres_placed < num_spheres:
            if ring == 0:
                # Center sphere
                if spheres_placed < num_spheres:
                    x = torch.tensor([0.0], device=device)
                    y = torch.tensor([0.0], device=device)
                    z = torch.tensor([z_pos], device=device)
                    center_pos = torch.stack((x, y, z), dim=1)
                    
                    # Check if center sphere fits
                    if torch.sqrt(x**2 + y**2) <= effective_radius:
                        positions = torch.cat((positions, center_pos), dim=0)
                        spheres_placed += 1
                ring += 1
                continue
            
            # Generate hexagonal ring
            spheres_in_ring = 6 * ring
            angles = torch.linspace(0, 2 * np.pi, spheres_in_ring + 1, device=device)[:-1]
            angles += layer_rotation  # Add layer rotation offset
            
            ring_radius = ring * sphere_spacing * 0.866  # Distance from center to ring
            
            x = ring_radius * torch.cos(angles)
            y = ring_radius * torch.sin(angles)
            
            # Check which spheres in this ring fit within the cylinder
            distances_from_center = torch.sqrt(x**2 + y**2)
            valid_mask = distances_from_center <= effective_radius
            valid_indices = torch.where(valid_mask)[0]
            
            # Take only as many as we need
            needed = min(len(valid_indices), num_spheres - spheres_placed)
            if needed > 0:
                selected_indices = valid_indices[:needed]
                x_valid = x[selected_indices]
                y_valid = y[selected_indices]
                z_valid = torch.full_like(x_valid, z_pos)
                
                ring_positions = torch.stack((x_valid, y_valid, z_valid), dim=1)
                positions = torch.cat((positions, ring_positions), dim=0)
                spheres_placed += needed
            
            ring += 1
            
            # Safety check: if ring radius exceeds cylinder, break
            if ring * sphere_spacing * 0.866 > effective_radius:
                break
        
        return positions


    def initialize_spheres_with_settling_torch(
        self,
        num_spheres, 
        sphere_radius, 
        cylinder_radius, 
        cylinder_height, 
        cylinder_thickness, 
        device="cuda"
    ):
        """
        Alternative approach: Initialize with slight randomization and apply settling physics.
        """
        # First get stable layered positions
        stable_positions = self.initialize_spheres_stable_torch(
            num_spheres, sphere_radius, cylinder_radius, 
            cylinder_height, cylinder_thickness, device=device
        )
        
        # Add small random perturbations to avoid perfect symmetry
        perturbation_scale = sphere_radius * 0.1
        perturbations = torch.randn_like(stable_positions, device=device) * perturbation_scale
        
        # Don't perturb z-coordinate much to maintain stability
        perturbations[:, 2] *= 0.1
        
        final_positions = stable_positions + perturbations
        
        # Ensure spheres are still within bounds after perturbation
        distances_from_center = torch.sqrt(final_positions[:, 0]**2 + final_positions[:, 1]**2)
        effective_radius = cylinder_radius - cylinder_thickness - sphere_radius
        
        # Clamp positions that went outside bounds
        outside_mask = distances_from_center > effective_radius
        if outside_mask.any():
            scale_factor = effective_radius / distances_from_center[outside_mask].unsqueeze(1)
            final_positions[outside_mask, :2] *= scale_factor
        
        # Ensure minimum z-coordinate
        min_z = sphere_radius + cylinder_thickness
        final_positions[:, 2] = torch.clamp(final_positions[:, 2], min=min_z)
        
        return final_positions


    #######################################
    ######### RealRobot Related ###########
    #######################################
    def pre_physics_step_real(self, actions):
        actions[:, -1] = 1. # Keep the gripper closing
        return super().pre_physics_step_real(actions)


    def _init_simulation_mode(self, cupA_pose, cupB_pose, franka_states):
        """Initialize simulation mode with real world cube poses."""
        # Convert poses to torch tensors
        cupA_quat, cupA_pos = [to_torch(x, device=self.device) for x in cupA_pose]
        cupB_quat, cupB_pos = [to_torch(x, device=self.device) for x in cupB_pose]
        
        # Get robot frame transformation
        w2franka_quat = self._franka_state[0, 3:7]
        w2franka_pos = self._franka_state[0, :3]
        
        # Transform cube poses to world frame
        # cupA_base_quat, cupA_base_pos = tf_combine(cupA_quat, cupA_pos, self.cupA_rim2base_quat[0], self.cupA_rim2base_pos[0])
        w2cup_quat, w2cup_pos = tf_combine(w2franka_quat, w2franka_pos, cupA_quat, cupA_pos)
        w2target_quat, w2target_pos = tf_combine(w2franka_quat, w2franka_pos, cupB_quat, cupB_pos)
        w2cup_pos[2] = torch.clamp(w2cup_pos[2], min=self._ws_surface_pos[2])
        w2target_pos[2] = torch.clamp(w2target_pos[2], min=self._ws_surface_pos[2])

        # Reset cube states
        w2cupA_pose = torch.cat([w2cup_pos, w2cup_quat], dim=-1).unsqueeze(0)
        w2cupB_pose = torch.cat([w2target_pos, w2target_quat], dim=-1).unsqueeze(0)
        self.reset_real_cup_states(w2cupA_pose, w2cupB_pose)
        
        self._update_states()
        # Validate robot state consistency if using FK replay
        if self.cfg.get("use_fk_replay") and franka_states is not None:
            self._validate_robot_state_consistency(franka_states)

        if hasattr(self, "env_configs") and self.env_configs is not None:
            estimated_time = self._estimate_minimum_time2end_real(self.env_configs, self.states)
            self._reset_timeaware_states_real(estimated_time)
        self.compute_observations()
        return self.obs_buf, self.extras
    

    def _compute_cupA_pose_from_franka(self, franka_states):
        """Compute cupA pose from franka end-effector pose."""
        if franka_states is None:
            eef_quat, eef_pos = self.states["eef_quat"][0], self.states["eef_pos"][0]
        else:
            eef_quat, eef_pos = to_torch(franka_states["eef_quat"]), to_torch(franka_states["eef_pos"])
        cupA_quat, cupA_pos = tf_combine(eef_quat, eef_pos, self.default_eef2cupA_quat[0], self.default_eef2cupA_pos[0])
        return [cupA_quat, cupA_pos]
    
    
    def _get_cup_poses_with_retry(self, state_estimator, max_attempts=1, init=False):
        """Get cube poses with retry logic."""
        cupB_pose_outdated = False
        for _ in range(max_attempts):
            state_estimator.update()
            # state_estimator.render(draw=True)
            cupB_pose = state_estimator.get_target_pose()
            if cupB_pose is not None:
                return cupB_pose, cupB_pose_outdated
        
        if init:
            raise ValueError(f"Failed to get cube poses after {max_attempts} attempts at initialization.")

        if cupB_pose is None:
            cupB_pose = state_estimator.get_target_last_pose()
            cupB_pose_outdated = True

        return cupB_pose, cupB_pose_outdated
    

    def init_real2sim(self, state_estimator, franka_arm):
        """Initialize the real-to-sim synchronization."""
        # Misc value initialize
        self.occlude_eef2cupA = None
        self.first_occlude_eef2cupA = None

        # Initial states computation
        cupB_pose, _ = self._get_cup_poses_with_retry(state_estimator, max_attempts=10, init=True)
        franka_states = self._get_robot_state_with_retry(franka_arm, max_attempts=500)
        cupA_pose = self._compute_cupA_pose_from_franka(franka_states)
        
        if self.cfg.get("use_sim_pure", False):
            return self._init_simulation_mode(cupA_pose, cupB_pose, franka_states)
        else:
            return self._init_real_robot_mode(state_estimator, franka_arm)


    def _update_task_states_real(self, state_estimator):
        """Update task-specific states (cube positions; All poses are in the franka base frame)."""
        cupB_pose, cupB_pose_outdated = self._get_cup_poses_with_retry(state_estimator)
        
        # Unpack poses
        cupB_quat, cupB_pos = cupB_pose
        cupB_rimquat, cupB_rimpos = tf_combine(to_torch(cupB_quat), to_torch(cupB_pos), 
                                               self.unit_quat.flatten(), self.cupB_base2rim.flatten())
        
        return {
            "cupB_pos": cupB_pos,
            "cupB_quat": cupB_quat,
            "cupB_rimpos": cupB_rimpos.cpu().numpy(),
            "cupB_rimquat": cupB_rimquat.cpu().numpy(),
            "cupB_pose_outdated": cupB_pose_outdated,
        }
    

    def _compensate_task_states_real(self, task_states, robot_states):
        # Must Compensate task states especially occlusion
        cupA_quat, cupA_pos = self._compute_cupA_pose_from_franka(robot_states)
        cupA_rimquat, cupA_rimpos = tf_combine(cupA_quat, cupA_pos, 
                                               self.unit_quat.flatten(), self.cupA_base2rim.flatten())
        task_states.update({
            "cupA_pos": cupA_pos.cpu().numpy(),
            "cupA_quat": cupA_quat.cpu().numpy(),
            "cupA_rimpos": cupA_rimpos.cpu().numpy(),
            "cupA_rimquat": cupA_rimquat.cpu().numpy(),
        })
        task_states.update({
            "cupA_to_cupB_pos": task_states["cupB_rimpos"] - task_states["cupA_rimpos"],
        })
        
        return task_states


    def _update_debug_info_real(self):
        """Update debug information from real robot states."""
        debug_info = {
            "eef_pos": self.states_real["eef_pos"],
            "eef_quat": self.states_real["eef_quat"],
            "cupA_pos": self.states_real["cupA_pos"],
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
            "cupA": {
                "pos": self.world_states["cupA_pos"][0].cpu().numpy(),
                "quat": self.world_states["cupA_quat"][0].cpu().numpy()
            },
            "cupB": {
                "pos": self.world_states["cupB_pos"][0].cpu().numpy(),
                "quat": self.world_states["cupB_quat"][0].cpu().numpy()
            }
        }
        
        # Draw coordinate axes for each cube
        self.clear_lines()
        for cube_name, pose in poses.items():
            self.draw_axes(pos=pose["pos"], ori=pose["quat"])


    #-------------Estimate the minimum time for the real new config with KNN-------------# 
    def _compute_init_config_features(self, data_dict):
        # Features: cupA_pos(3), cupB_pos(3), franka_dof(9)
        features = torch.cat([data_dict["cupA_state"][:, :3], 
                              data_dict["cupB_state"][:, :3], 
                              data_dict["franka_state"][:, :7]], dim=-1)
        return features

    
    def _time_related_state_names(self):
        return ["cupA_pos", "cupB_pos", "q"]


#####################################################################
###=========================jit functions=========================###
#####################################################################
# Weird error, the torch.jit.script with rewards seem introduce cycling tensor and introducing gpu memory leak problem
# rewards can not be memorized in any way such as self.rew_buf = rewards, or reward_buf[step] = rewards ... Weird, only to rewards
# However, the rl_games can bypass this problem and keeps using torch.jit.script. Fix later