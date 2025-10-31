import os
from os.path import join
import json
import time
from datetime import datetime
from typing import Dict, Any, Tuple, List, Set

import gym
from gym import spaces

from isaacgym import gymtorch, gymapi, gymutil
from envs.isaacgymenvs.utils.torch_jit_utils import to_torch, torch_rand_float, torch_rand_float_3d, tf_combine, tf_apply, tf_inverse, quat_apply, normalize, quat_from_euler_xyz, quat_mul, tensor_clamp, axisangle2quat
from envs.isaacgymenvs.utils.dr_utils import get_property_setter_map, get_property_getter_map, \
    get_default_setter_args, apply_random_samples, check_buckets, generate_random_samples
from tf_utils import quaternion_distance

import torch
import numpy as np
import operator, random
from copy import deepcopy
from sklearn.preprocessing import StandardScaler

from collections import deque

import sys

import abc
from abc import ABC

EXISTING_SIM = None
SCREEN_CAPTURE_RESOLUTION = (1027, 768)

def _create_sim_once(gym, *args, **kwargs):
    global EXISTING_SIM
    if EXISTING_SIM is not None:
        return EXISTING_SIM
    else:
        EXISTING_SIM = gym.create_sim(*args, **kwargs)
        return EXISTING_SIM


class Env(ABC):
    def __init__(self, config: Dict[str, Any], rl_device: str, sim_device: str, graphics_device_id: int, headless: bool): 
        """Initialise the env.

        Args:
            config: the configuration dictionary.
            sim_device: the device to simulate physics on. eg. 'cuda:0' or 'cpu'
            graphics_device_id: the device ID to render with.
            headless: Set to False to disable viewer rendering.
        """

        split_device = sim_device.split(":")
        self.device_type = split_device[0]
        self.device_id = int(split_device[1]) if len(split_device) > 1 else 0

        self.device = "cpu"
        if config["sim"]["use_gpu_pipeline"]:
            if self.device_type.lower() == "cuda" or self.device_type.lower() == "gpu":
                self.device = "cuda" + ":" + str(self.device_id)
            else:
                print("GPU Pipeline can only be used with GPU simulation. Forcing CPU Pipeline.")
                config["sim"]["use_gpu_pipeline"] = False

        self.rl_device = rl_device

        # Rendering
        # if training in a headless mode
        self.headless = headless

        enable_camera_sensors = config["env"].get("enableCameraSensors", False)
        self.graphics_device_id = graphics_device_id
        if enable_camera_sensors == False and self.headless == True:
            self.graphics_device_id = -1

        self.num_environments = config["env"]["numEnvs"]
        self.num_agents = config["env"].get("numAgents", 1)  # used for multi-agent environments

        self.num_observations = config["env"].get("numObservations")
        self.num_actions = config["env"].get("numActions")
        self.num_states = config["env"].get("numStates", 0)

        self.obs_space = spaces.Box(np.ones(self.num_obs) * -np.Inf, np.ones(self.num_obs) * np.Inf)
        self.state_space = spaces.Box(np.ones(self.num_states) * -np.Inf, np.ones(self.num_states) * np.Inf)
        self.act_space = spaces.Box(np.ones(self.num_actions) * -1., np.ones(self.num_actions) * 1.)

        self.clip_obs = config["env"].get("clipObservations", np.Inf)
        self.clip_actions = config["env"].get("clipActions", np.Inf)

        # Total number of training frames since the beginning of the experiment.
        # We get this information from the learning algorithm rather than tracking ourselves.
        # The learning algorithm tracks the total number of frames since the beginning of training and accounts for
        # experiments restart/resumes. This means this number can be > 0 right after initialization if we resume the
        # experiment.
        self.total_train_env_frames: int = 0

        # number of control steps
        self.control_steps: int = 0

        self.render_fps: int = config["env"].get("renderFPS", -1)
        self.last_frame_time: float = 0.0

        self.record_frames: bool = False
        self.record_frames_dir = join("recorded_frames", datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))

    @abc.abstractmethod 
    def allocate_buffers(self):
        """Create torch buffers for observations, rewards, actions dones and any additional data."""

    @abc.abstractmethod
    def step(self, actions: torch.Tensor) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """Step the physics of the environment.
        Args:
            actions: actions to apply
        Returns:
            Observations, rewards, resets, info
            Observations are dict of observations (currently only one member called 'obs')
        """

    @abc.abstractmethod
    def reset(self)-> Dict[str, torch.Tensor]:
        """Reset the environment.
        Returns:
            Observation dictionary
        """

    @abc.abstractmethod
    def reset_idx(self, env_ids: torch.Tensor):
        """Reset environments having the provided indices.
        Args:
            env_ids: environments to reset
        """

    @property
    def observation_space(self) -> gym.Space:
        """Get the environment's observation space."""
        return self.obs_space

    @property
    def action_space(self) -> gym.Space:
        """Get the environment's action space."""
        return self.act_space

    @property
    def num_envs(self) -> int:
        """Get the number of environments."""
        return self.num_environments

    @property
    def num_acts(self) -> int:
        """Get the number of actions in the environment."""
        return self.num_actions

    @property
    def num_obs(self) -> int:
        """Get the number of observations in the environment."""
        return self.num_observations
    
    @property
    def num_state(self) -> int:
        """Get the number of states in the environment."""
        return self.num_states

    def set_train_info(self, env_frames, *args, **kwargs):
        """
        Send the information in the direction algo->environment.
        Most common use case: tell the environment how far along we are in the training process. This is useful
        for implementing curriculums and things such as that.
        """
        self.total_train_env_frames = env_frames
        # print(f'env_frames updated to {self.total_train_env_frames}')

    def get_env_state(self):
        """
        Return serializable environment state to be saved to checkpoint.
        Can be used for stateful training sessions, i.e. with adaptive curriculums.
        """
        return None

    def set_env_state(self, env_state):
        pass


class VecTask(Env):

    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 24}

    def __init__(self, config, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture: bool = False, force_render: bool = False): 
        """Initialise the `VecTask`.

        Args:
            config: config dictionary for the environment.
            sim_device: the device to simulate physics on. eg. 'cuda:0' or 'cpu'
            graphics_device_id: the device ID to render with.
            headless: Set to False to disable viewer rendering.
            virtual_screen_capture: Set to True to allow the users get captured screen in RGB array via `env.render(mode='rgb_array')`. 
            force_render: Set to True to always force rendering in the steps (if the `control_freq_inv` is greater than 1 we suggest stting this arg to True)
        """
        # super().__init__(config, rl_device, sim_device, graphics_device_id, headless, use_dict_obs)
        super().__init__(config, rl_device, sim_device, graphics_device_id, headless)
        self.virtual_screen_capture = virtual_screen_capture
        self.virtual_display = None
        if self.virtual_screen_capture:
            from pyvirtualdisplay.smartdisplay import SmartDisplay
            self.virtual_display = SmartDisplay(size=SCREEN_CAPTURE_RESOLUTION)
            self.virtual_display.start()
        self.force_render = force_render

        self.sim_params = self.__parse_sim_params(self.cfg["physics_engine"], self.cfg["sim"])
        if self.cfg["physics_engine"] == "physx":
            self.physics_engine = gymapi.SIM_PHYSX
        elif self.cfg["physics_engine"] == "flex":
            self.physics_engine = gymapi.SIM_FLEX
        else:
            msg = f"Invalid physics engine backend: {self.cfg['physics_engine']}"
            raise ValueError(msg)

        # optimization flags for pytorch JIT
        torch._C._jit_set_profiling_mode(False)
        torch._C._jit_set_profiling_executor(False)

        self.gym = gymapi.acquire_gym()

        self.first_randomization = True
        self.original_props = {}
        self.dr_randomizations = {}
        self.actor_params_generator = None
        self.extern_actor_params = {}
        self.last_step = -1
        self.last_rand_step = -1
        for env_id in range(self.num_envs):
            self.extern_actor_params[env_id] = None

        # create envs, sim and viewer
        self.sim_initialized = False
        self.create_sim()
        self.gym.prepare_sim(self.sim)
        self.sim_initialized = True

        self.set_viewer()
        self.allocate_buffers()

        self.obs_dict = {}

        self.init_cur_dr_params(init_curri_ratio=self.cfg.get("init_curri_ratio", 0.0))

    def set_viewer(self):
        """Create the viewer."""

        # todo: read from config
        self.enable_viewer_sync = True
        self.viewer = None
        self.camera = None
        camera_props = gymapi.CameraProperties()
        self.camera_w, self.camera_h = camera_props.width, camera_props.height

        # if running with a viewer, set up keyboard shortcuts and camera
        if self.headless == False:
            # subscribe to keyboard shortcuts
            self.viewer = self.gym.create_viewer(
                self.sim, gymapi.CameraProperties())
            self.camera = self.gym.create_camera_sensor(
                self.envs[0], gymapi.CameraProperties())
            
            self.gym.subscribe_viewer_keyboard_event(
                self.viewer, gymapi.KEY_ESCAPE, "QUIT")
            self.gym.subscribe_viewer_keyboard_event(
                self.viewer, gymapi.KEY_V, "toggle_viewer_sync")
            self.gym.subscribe_viewer_keyboard_event(
                self.viewer, gymapi.KEY_R, "record_frames")

            # set the camera position based on up axis
            sim_params = self.gym.get_sim_params(self.sim)
            if self.cfg.get("task_name", "") == "FrankaCubeStack":
                cam_pos = gymapi.Vec3(0.9, 0., 1.7)
                cam_target = gymapi.Vec3(-2.5, 0., 0.0)
            elif self.cfg.get("task_name", "") == "FrankaGmPour":
                cam_pos = gymapi.Vec3(0.5, 0., 1.2)
                cam_target = gymapi.Vec3(-5, 0., 0.0)
            elif self.cfg.get("task_name", "") == "FrankaCabinet":
                cam_pos = gymapi.Vec3(-0.2, -0.8, 1.8)
                cam_target = gymapi.Vec3(1, 2.5, 0.0)
            else:
                cam_pos = gymapi.Vec3(0.9, 0., 1.7)
                cam_target = gymapi.Vec3(-2.5, 0., 0.0)

            self.gym.viewer_camera_look_at(
                self.viewer, None, cam_pos, cam_target)
            self.gym.set_camera_location(
                self.camera, self.envs[0], cam_pos, cam_target)


    def allocate_buffers(self):
        """Allocate the observation, states, etc. buffers.

        These are what is used to set observations and states in the environment classes which
        inherit from this one, and are read in `step` and other related functions.

        """
        self.extras = {}
        self.state_slice = {}
        self.obs_slice = {}
        # allocate buffers
        self.state_queue = [torch.zeros(
            (self.num_envs, self.numSingleState), device=self.device, dtype=torch.float)] * self.sequence_len
        self.obs_queue = [torch.zeros(
            (self.num_envs, self.numSingleObs), device=self.device, dtype=torch.float)] * self.sequence_len
        self.states_buf = torch.zeros(
            (self.num_envs, self.num_states), device=self.device, dtype=torch.float)
        self.obs_buf = torch.zeros(
            (self.num_envs, self.num_observations), device=self.device, dtype=torch.float)
        
        self.fixed_state_queue = [torch.zeros(
            (self.num_envs, self.numSingleState), device=self.device, dtype=torch.float)] * self.sequence_len
        self.fixed_obs_queue = [torch.zeros(
            (self.num_envs, self.numSingleObs), device=self.device, dtype=torch.float)] * self.sequence_len
        self.fixed_state_buf = torch.zeros(
            (self.num_envs, self.numSingleState), device=self.device, dtype=torch.float)
        self.fixed_obs_buf = torch.zeros(
            (self.num_envs, self.numSingleObs), device=self.device, dtype=torch.float)
        
        self.prev_state = torch.zeros(
            (self.num_envs, self.numSingleState), device=self.device, dtype=torch.float)
        self.prev_obs = torch.zeros(
            (self.num_envs, self.numSingleObs), device=self.device, dtype=torch.float)
        self.prev_tgtq = torch.zeros(
            (self.num_envs, self.num_franka_dofs-2), device=self.device, dtype=torch.float)
        self.prev_tgtq_gripper = torch.zeros(
            (self.num_envs, 2), device=self.device, dtype=torch.float)
        self.prev_dq = torch.zeros(
            (self.num_envs, self.num_franka_dofs-2), device=self.device, dtype=torch.float)
        
        self.rew_buf = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.float)
        self.reset_buf = torch.ones(
            self.num_envs, device=self.device, dtype=torch.long)
        self.success_buf = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.long)
        self.timeout_buf = torch.zeros(
             self.num_envs, device=self.device, dtype=torch.long)
        self.progress_buf = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.long)
        self.randomize_buf = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.long)
        self.continuous_check_buf = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.long)
        self.force_has_applied = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.long)
        # Custom. time_cur_buf is used to track the time for data collection
        self.time_ratio_buf = torch.ones(
            self.num_envs, device=self.device, dtype=torch.float)
        self.prev_time_ratio_buf = torch.ones(
            self.num_envs, device=self.device, dtype=torch.float)
        self.time_cur_buf = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.float)
        self.time_to_end = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.float)
        self.time2end_init = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.float)
        self.real_time_to_end = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.float)
        self.real_time2end_init = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.float)
        self.interaction_time = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.float)
        self.max_linvel_buf = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.float)
        self.sce_linvel_buf = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.float)
        self.sce_linacc_buf = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.float)
        self.linvel_max_gt = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.float)
        self.linvel_max_gt_init = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.float)
        self.accm_instability = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.float)
        self.env2index = -torch.ones(
            self.num_envs, device=self.device, dtype=torch.long)
        # Realworld Buffer
        self.real_obs_queue = [torch.zeros(
            (1, self.numSingleObs), device=self.device, dtype=torch.float)] * self.sequence_len


    def create_sim(self, compute_device: int, graphics_device: int, physics_engine, sim_params: gymapi.SimParams):
        """Create an Isaac Gym sim object.

        Args:
            compute_device: ID of compute device to use.
            graphics_device: ID of graphics device to use.
            physics_engine: physics engine to use (`gymapi.SIM_PHYSX` or `gymapi.SIM_FLEX`)
            sim_params: sim params to use.
        Returns:
            the Isaac Gym sim object.
        """
        sim = _create_sim_once(self.gym, compute_device, graphics_device, physics_engine, sim_params)
        if sim is None:
            print("*** Failed to create sim")
            quit()

        return sim

    def get_state(self):
        """Returns the state buffer of the environment (the privileged observations for asymmetric training)."""
        return torch.clamp(self.states_buf, -self.clip_obs, self.clip_obs).to(self.rl_device)
    

    @abc.abstractmethod
    def debug_viz(self):
        """Debug visualization for the environment. This is called at the end of each step."""
    

    def pre_physics_step(self, actions: torch.Tensor):
        """Apply the actions to the environment (eg by setting torques, position targets).

        Args:
            actions: the actions to apply
        """
        self.convert_actions(actions)
        self.command_arm()
        self.command_gripper()


    def post_physics_step(self):
        """Compute reward and observations, reset any environments that require it."""
        # Reset envs and apply initial states MUST after refresh because refresh will overwrite the states to the stale one which has been reset
        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        success_env_ids = self.success_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) > 0 and not self.cfg.get("real_robot", False):
            if self.is_ready_to_record(success_env_ids):
                self.record_post_config(success_env_ids)
            self.reset_idx(env_ids)
        
        # Refresh states usually should be after recording the previous states and before resetting the envs (setter function)! 
        # However, we include an extra simulate step in reset to make sure the states are updated.
        self._refresh()

        self.compute_observations(reset_ids=env_ids)
        self.compute_reward(self.actions)

        if not self.training:
            if self.cfg.get("apply_disturbances", False):
                self.apply_disturbances()
            if self.cfg.get("vis_configs", False):
                self.visualize_configs_seq()

        if self.viewer and self.debug_vis:
            self.debug_viz()

    def step(self, actions: torch.Tensor) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """Step the physics of the environment.

        Args:
            actions: actions to apply
        Returns:
            Observations, rewards, resets, info
            Observations are dict of observations (currently only one member called 'obs')
        """

        # randomize actions
        if self.dr_randomizations.get('actions', None):
            actions = self.dr_randomizations['actions']['noise_lambda'](actions)

        action_tensor = torch.clamp(actions, -self.clip_actions, self.clip_actions)
        # apply actions
        self.pre_physics_step(action_tensor)

        # Interpolate joint positions from the current joint positions to the target joint positions
        for i in range(self.num_inter_steps):
            self.deploy_joint_command(i)
            # step physics and render each frame
            for j in range(self.control_freq_inv):
                if self.force_render:
                    self.render()
                    # render the frame using cv2
                    # self.last_frame = self.get_viewer_image()
                    # print(self.last_frame.shape, self.last_frame.dtype)
                    # render_frame = cv2.cvtColor(self.last_frame, cv2.COLOR_RGBA2BGR)
                    # cv2.imshow('frame', render_frame)
                    # cv2.waitKey(1)
                self.gym.simulate(self.sim)
                self.progress_buf += 1

        # to fix!
        if self.device == 'cpu':
            self.gym.fetch_results(self.sim, True)

        # compute observations, rewards, resets, ...
        self.post_physics_step()

        self.control_steps += 1
        self.policy_steps = self.progress_buf // (self.num_inter_steps * self.control_freq_inv)

        # fill time out buffer: set to 1 if we reached the max episode length AND the reset buffer is 1. Timeout == 1 makes sense only if the reset buffer is 1.
        self.timeout_buf = (self.progress_buf >= self.max_episode_length - 1) & (self.reset_buf != 0)

        self.extras["time_outs"] = self.timeout_buf.to(self.rl_device)
        self.extras["success"] = self.success_buf.to(self.rl_device)

        self.update_observations_dict()

        return self.obs_dict, self.rew_buf.to(self.rl_device), self.reset_buf.to(self.rl_device), self.extras

    def zero_actions(self) -> torch.Tensor:
        """Returns a buffer with zero actions.

        Returns:
            A buffer of zero torch actions
        """
        actions = torch.zeros([self.num_envs, self.num_actions], dtype=torch.float32, device=self.rl_device)

        return actions

    def reset_all(self):
        reset_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        self.success_counts[reset_ids] = self.save_threshold # Manually reset configs for new init recording
        self._update_props(reset_ids)
        self.reset_idx(reset_ids)
        self._refresh()
        self.compute_observations(reset_ids=reset_ids)
    
    def reset_idx(self, env_idx):
        """Reset environment with indces in env_idx. 
        Should be implemented in an environment class inherited from VecTask.
        """  
        pass

    def reset(self):
        """Is called only once when environment starts to provide the first observations.
        Doesn't calculate observations. Actual reset and observation calculation need to be implemented by user.
        Returns:
            Observation dictionary
        """
        self.update_observations_dict()

        return self.obs_dict

    def reset_done(self):
        """Reset the environment.
        Returns:
            Observation dictionary, indices of environments being reset
        """
        done_env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        if len(done_env_ids) > 0:
            self.reset_idx(done_env_ids)

        self.update_observations_dict()

        return self.obs_dict, done_env_ids
    
    def _refresh(self):
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_jacobian_tensors(self.sim)
        self.gym.refresh_mass_matrix_tensors(self.sim)
        if self.force_render:
            self.gym.render_all_camera_sensors(self.sim)

    def render(self, mode="rgb_array"):
        """Draw the frame to the viewer, and check for keyboard events."""
        if self.viewer:
            # check for window closed
            if self.gym.query_viewer_has_closed(self.viewer):
                sys.exit()

            # check for keyboard events
            for evt in self.gym.query_viewer_action_events(self.viewer):
                if evt.action == "QUIT" and evt.value > 0:
                    sys.exit()
                elif evt.action == "toggle_viewer_sync" and evt.value > 0:
                    self.enable_viewer_sync = not self.enable_viewer_sync
                elif evt.action == "record_frames" and evt.value > 0:
                    self.record_frames = not self.record_frames

            # fetch results
            if self.device != 'cpu':
                self.gym.fetch_results(self.sim, True)

            # step graphics
            if self.enable_viewer_sync:
                self.gym.step_graphics(self.sim)
                self.gym.draw_viewer(self.viewer, self.sim, True)

                # Wait for dt to elapse in real time.
                # This synchronizes the physics simulation with the rendering rate.
                self.gym.sync_frame_time(self.sim)

                # it seems like in some cases sync_frame_time still results in higher-than-realtime framerate
                # this code will slow down the rendering to real time
                now = time.perf_counter()
                delta = now - self.last_frame_time
                if self.render_fps < 0:
                    # render at control frequency
                    render_dt = self.dt
                else:
                    render_dt = 1.0 / self.render_fps

                if delta < render_dt:
                    time.sleep(render_dt - delta)

                self.last_frame_time = time.perf_counter()

            else:
                self.gym.poll_viewer_events(self.viewer)

            if self.record_frames:
                if not os.path.isdir(self.record_frames_dir):
                    os.makedirs(self.record_frames_dir, exist_ok=True)

                self.gym.write_viewer_image_to_file(self.viewer, join(self.record_frames_dir, f"frame_{self.control_steps}.png"))

            if self.virtual_display and mode == "rgb_array":
                img = self.virtual_display.grab()
                return np.array(img)

    def __parse_sim_params(self, physics_engine: str, config_sim: Dict[str, Any]) -> gymapi.SimParams:
        """Parse the config dictionary for physics stepping settings.

        Args:
            physics_engine: which physics engine to use. "physx" or "flex"
            config_sim: dict of sim configuration parameters
        Returns
            IsaacGym SimParams object with updated settings.
        """
        sim_params = gymapi.SimParams()

        # check correct up-axis
        if config_sim["up_axis"] not in ["z", "y"]:
            msg = f"Invalid physics up-axis: {config_sim['up_axis']}"
            print(msg)
            raise ValueError(msg)

        # assign general sim parameters
        sim_params.dt = self.cfg.get("dt", 1/60)
        sim_params.num_client_threads = config_sim.get("num_client_threads", 0)
        sim_params.use_gpu_pipeline = config_sim["use_gpu_pipeline"]
        sim_params.substeps = config_sim.get("substeps", 2)

        # assign up-axis
        if config_sim["up_axis"] == "z":
            sim_params.up_axis = gymapi.UP_AXIS_Z
        else:
            sim_params.up_axis = gymapi.UP_AXIS_Y

        # assign gravity
        sim_params.gravity = gymapi.Vec3(*config_sim["gravity"])

        # configure physics parameters
        if physics_engine == "physx":
            # set the parameters
            if "physx" in config_sim:
                for opt in config_sim["physx"].keys():
                    if opt == "contact_collection":
                        setattr(sim_params.physx, opt, gymapi.ContactCollection(config_sim["physx"][opt]))
                    else:
                        setattr(sim_params.physx, opt, config_sim["physx"][opt])
        else:
            # set the parameters
            if "flex" in config_sim:
                for opt in config_sim["flex"].keys():
                    setattr(sim_params.flex, opt, config_sim["flex"][opt])

        # return the configured params
        return sim_params

    """
    Domain Randomization methods
    """

    def get_actor_params_info(self, dr_params: Dict[str, Any], env):
        """Generate a flat array of actor params, their names and ranges.

        Returns:
            The array
        """

        if "actor_params" not in dr_params:
            return None
        params = []
        names = []
        lows = []
        highs = []
        param_getters_map = get_property_getter_map(self.gym)
        for actor, actor_properties in dr_params["actor_params"].items():
            handle = self.gym.find_actor_handle(env, actor)
            for prop_name, prop_attrs in actor_properties.items():
                if prop_name == 'color':
                    continue  # this is set randomly
                props = param_getters_map[prop_name](env, handle)
                if not isinstance(props, list):
                    props = [props]
                for prop_idx, prop in enumerate(props):
                    for attr, attr_randomization_params in prop_attrs.items():
                        name = prop_name+'_' + str(prop_idx) + '_'+attr
                        lo_hi = attr_randomization_params['range']
                        distr = attr_randomization_params['distribution']
                        if 'uniform' not in distr:
                            lo_hi = (-1.0*float('Inf'), float('Inf'))
                        if isinstance(prop, np.ndarray):
                            for attr_idx in range(prop[attr].shape[0]):
                                params.append(prop[attr][attr_idx])
                                names.append(name+'_'+str(attr_idx))
                                lows.append(lo_hi[0])
                                highs.append(lo_hi[1])
                        else:
                            params.append(getattr(prop, attr))
                            names.append(name)
                            lows.append(lo_hi[0])
                            highs.append(lo_hi[1])
        return params, names, lows, highs

    def apply_randomizations(self, dr_params):
        """Apply domain randomizations to the environment.

        Note that currently we can only apply randomizations only on resets, due to current PhysX limitations

        Args:
            dr_params: parameters for domain randomization to use.
        """

        # If we don't have a randomization frequency, randomize every step
        rand_freq = dr_params.get("frequency", 1)

        # First, determine what to randomize:
        #   - non-environment parameters when > frequency steps have passed since the last non-environment
        #   - physical environments in the reset buffer, which have exceeded the randomization frequency threshold
        #   - on the first call, randomize everything
        self.last_step = self.gym.get_frame_count(self.sim)
        if self.first_randomization:
            do_nonenv_randomize = True
            env_ids = list(range(self.num_envs))
        else:
            do_nonenv_randomize = (self.last_step - self.last_rand_step) >= rand_freq
            rand_envs = torch.where(self.randomize_buf >= rand_freq, torch.ones_like(self.randomize_buf), torch.zeros_like(self.randomize_buf))
            rand_envs = torch.logical_and(rand_envs, self.reset_buf)
            env_ids = torch.nonzero(rand_envs, as_tuple=False).squeeze(-1).tolist()
            self.randomize_buf[rand_envs] = 0

        if do_nonenv_randomize:
            self.last_rand_step = self.last_step

        param_setters_map = get_property_setter_map(self.gym)
        param_setter_defaults_map = get_default_setter_args(self.gym)
        param_getters_map = get_property_getter_map(self.gym)

        # On first iteration, check the number of buckets
        if self.first_randomization:
            check_buckets(self.gym, self.envs, dr_params)

        for nonphysical_param in ["observations", "actions"]:
            if nonphysical_param in dr_params and do_nonenv_randomize:
                dist = dr_params[nonphysical_param]["distribution"]
                op_type = dr_params[nonphysical_param]["operation"]
                sched_type = dr_params[nonphysical_param]["schedule"] if "schedule" in dr_params[nonphysical_param] else None
                sched_step = dr_params[nonphysical_param]["schedule_steps"] if "schedule" in dr_params[nonphysical_param] else None
                op = operator.add if op_type == 'additive' else operator.mul

                if sched_type == 'linear':
                    sched_scaling = 1.0 / sched_step * \
                        min(self.last_step, sched_step)
                elif sched_type == 'constant':
                    sched_scaling = 0 if self.last_step < sched_step else 1
                else:
                    sched_scaling = 1

                if dist == 'gaussian':
                    mu, var = dr_params[nonphysical_param]["range"]
                    mu_corr, var_corr = dr_params[nonphysical_param].get("range_correlated", [0., 0.])

                    if op_type == 'additive':
                        mu *= sched_scaling
                        var *= sched_scaling
                        mu_corr *= sched_scaling
                        var_corr *= sched_scaling
                    elif op_type == 'scaling':
                        var = var * sched_scaling  # scale up var over time
                        mu = mu * sched_scaling + 1.0 * \
                            (1.0 - sched_scaling)  # linearly interpolate

                        var_corr = var_corr * sched_scaling  # scale up var over time
                        mu_corr = mu_corr * sched_scaling + 1.0 * \
                            (1.0 - sched_scaling)  # linearly interpolate

                    def noise_lambda(tensor, param_name=nonphysical_param):
                        params = self.dr_randomizations[param_name]
                        corr = params.get('corr', None)
                        if corr is None:
                            corr = torch.randn_like(tensor)
                            params['corr'] = corr
                        corr = corr * params['var_corr'] + params['mu_corr']
                        return op(
                            tensor, corr + torch.randn_like(tensor) * params['var'] + params['mu'])

                    self.dr_randomizations[nonphysical_param] = {'mu': mu, 'var': var, 'mu_corr': mu_corr, 'var_corr': var_corr, 'noise_lambda': noise_lambda}

                elif dist == 'uniform':
                    lo, hi = dr_params[nonphysical_param]["range"]
                    lo_corr, hi_corr = dr_params[nonphysical_param].get("range_correlated", [0., 0.])

                    if op_type == 'additive':
                        lo *= sched_scaling
                        hi *= sched_scaling
                        lo_corr *= sched_scaling
                        hi_corr *= sched_scaling
                    elif op_type == 'scaling':
                        lo = lo * sched_scaling + 1.0 * (1.0 - sched_scaling)
                        hi = hi * sched_scaling + 1.0 * (1.0 - sched_scaling)
                        lo_corr = lo_corr * sched_scaling + 1.0 * (1.0 - sched_scaling)
                        hi_corr = hi_corr * sched_scaling + 1.0 * (1.0 - sched_scaling)

                    def noise_lambda(tensor, param_name=nonphysical_param):
                        params = self.dr_randomizations[param_name]
                        corr = params.get('corr', None)
                        if corr is None:
                            corr = torch.randn_like(tensor)
                            params['corr'] = corr
                        corr = corr * (params['hi_corr'] - params['lo_corr']) + params['lo_corr']
                        return op(tensor, corr + torch.rand_like(tensor) * (params['hi'] - params['lo']) + params['lo'])

                    self.dr_randomizations[nonphysical_param] = {'lo': lo, 'hi': hi, 'lo_corr': lo_corr, 'hi_corr': hi_corr, 'noise_lambda': noise_lambda}

        if "sim_params" in dr_params and do_nonenv_randomize:
            prop_attrs = dr_params["sim_params"]
            prop = self.gym.get_sim_params(self.sim)

            if self.first_randomization:
                self.original_props["sim_params"] = {
                    attr: getattr(prop, attr) for attr in dir(prop)}

            for attr, attr_randomization_params in prop_attrs.items():
                apply_random_samples(
                    prop, self.original_props["sim_params"], attr, attr_randomization_params, self.last_step)

            self.gym.set_sim_params(self.sim, prop)

        # If self.actor_params_generator is initialized: use it to
        # sample actor simulation params. This gives users the
        # freedom to generate samples from arbitrary distributions,
        # e.g. use full-covariance distributions instead of the DR's
        # default of treating each simulation parameter independently.
        extern_offsets = {}
        if self.actor_params_generator is not None:
            for env_id in env_ids:
                self.extern_actor_params[env_id] = \
                    self.actor_params_generator.sample()
                extern_offsets[env_id] = 0

        # randomise all attributes of each actor (hand, cube etc..)
        # actor_properties are (stiffness, damping etc..)

        # Loop over actors, then loop over envs, then loop over their props 
        # and lastly loop over the ranges of the params 

        for actor, actor_properties in dr_params["actor_params"].items():

            # Loop over all envs as this part is not tensorised yet 
            for env_id in env_ids:
                env = self.envs[env_id]
                handle = self.gym.find_actor_handle(env, actor)
                extern_sample = self.extern_actor_params[env_id]

                # randomise dof_props, rigid_body, rigid_shape properties 
                # all obtained from the YAML file
                # EXAMPLE: prop name: dof_properties, rigid_body_properties, rigid_shape properties  
                #          prop_attrs: 
                #               {'damping': {'range': [0.3, 3.0], 'operation': 'scaling', 'distribution': 'loguniform'}
                #               {'stiffness': {'range': [0.75, 1.5], 'operation': 'scaling', 'distribution': 'loguniform'}
                for prop_name, prop_attrs in actor_properties.items():
                    if prop_name == 'color':
                        num_bodies = self.gym.get_actor_rigid_body_count(
                            env, handle)
                        for n in range(num_bodies):
                            self.gym.set_rigid_body_color(env, handle, n, gymapi.MESH_VISUAL,
                                                          gymapi.Vec3(random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1)))
                        continue

                    if prop_name == 'scale':
                        setup_only = prop_attrs.get('setup_only', False)
                        if (setup_only and not self.sim_initialized) or not setup_only:
                            attr_randomization_params = prop_attrs
                            sample = generate_random_samples(attr_randomization_params, 1,
                                                             self.last_step, None)
                            og_scale = 1
                            if attr_randomization_params['operation'] == 'scaling':
                                new_scale = og_scale * sample
                            elif attr_randomization_params['operation'] == 'additive':
                                new_scale = og_scale + sample
                            self.gym.set_actor_scale(env, handle, new_scale)
                        continue

                    prop = param_getters_map[prop_name](env, handle)
                    set_random_properties = True

                    if isinstance(prop, list):
                        if self.first_randomization:
                            self.original_props[prop_name] = [
                                {attr: getattr(p, attr) for attr in dir(p)} for p in prop]
                        for p, og_p in zip(prop, self.original_props[prop_name]):
                            for attr, attr_randomization_params in prop_attrs.items():
                                setup_only = attr_randomization_params.get('setup_only', False)
                                if (setup_only and not self.sim_initialized) or not setup_only:
                                    smpl = None
                                    if self.actor_params_generator is not None:
                                        smpl, extern_offsets[env_id] = get_attr_val_from_sample(
                                            extern_sample, extern_offsets[env_id], p, attr)
                                    apply_random_samples(
                                        p, og_p, attr, attr_randomization_params,
                                        self.last_step, smpl)
                                else:
                                    set_random_properties = False
                    else:
                        if self.first_randomization:
                            self.original_props[prop_name] = deepcopy(prop)
                        for attr, attr_randomization_params in prop_attrs.items():
                            setup_only = attr_randomization_params.get('setup_only', False)
                            if (setup_only and not self.sim_initialized) or not setup_only:
                                smpl = None
                                if self.actor_params_generator is not None:
                                    smpl, extern_offsets[env_id] = get_attr_val_from_sample(
                                        extern_sample, extern_offsets[env_id], prop, attr)
                                apply_random_samples(
                                    prop, self.original_props[prop_name], attr,
                                    attr_randomization_params, self.last_step, smpl)
                            else:
                                set_random_properties = False

                    if set_random_properties:
                        setter = param_setters_map[prop_name]
                        default_args = param_setter_defaults_map[prop_name]
                        setter(env, handle, prop, *default_args)

        if self.actor_params_generator is not None:
            for env_id in env_ids:  # check that we used all dims in sample
                if extern_offsets[env_id] > 0:
                    extern_sample = self.extern_actor_params[env_id]
                    if extern_offsets[env_id] != extern_sample.shape[0]:
                        print('env_id', env_id,
                              'extern_offset', extern_offsets[env_id],
                              'vs extern_sample.shape', extern_sample.shape)
                        raise Exception("Invalid extern_sample size")

        self.first_randomization = False


    
    def update_dr_params(self, curri_ratio=0.):
        self.cur_curri_ratio = curri_ratio
        for (sub_domain_settings, sub_domain_cur_params) in zip(self.dr_settings.values(), self.cur_dr_params.values()):
            
            for key, dr_range in sub_domain_settings.items():
                sub_domain_cur_params[key] = dr_range[0] + curri_ratio * (dr_range[1] - dr_range[0])
                sub_domain_cur_params[key] = max(min(sub_domain_cur_params[key], dr_range[1]), dr_range[0])

        # Update all related parameters using cur_dr_params
        self.update_max_velocity()
            

    def curriculum_noise(self, dr_params: Dict[str, Any], env):
        pass


    def curriculum_properties_dr(self, dr_params: Dict[str, Any], env):
        pass


    def curriculum_ctrl_freq(self, dr_params: Dict[str, Any], env):
        pass


    def init_cur_dr_params(self, init_curri_ratio=0.):
        """Get the current domain randomization parameters."""
        self.cur_curri_ratio = init_curri_ratio
        
        for domain_name, domain_dr_settings in self.dr_settings.items():
            for domain_key, dr_range in domain_dr_settings.items():
                if self.training:
                    self.cur_dr_params[domain_name][domain_key] = dr_range[0] + init_curri_ratio * (dr_range[1] - dr_range[0])
                else:
                    # Evaluation we only apply spatial randomization
                    if self.cfg.get("apply_noise_eval", False) or \
                       domain_name == "spatial" or domain_name == "controller":
                        self.cur_dr_params[domain_name][domain_key] = dr_range[0] + init_curri_ratio * (dr_range[1] - dr_range[0])
                    else:
                        self.cur_dr_params[domain_name][domain_key] = dr_range[0]

        # Update all related parameters using cur_dr_params
        self.update_max_velocity()


    ##############################
    ######### Custom #############
    ##############################
    def _create_franka_assets(self, asset_root):
        franka_asset_file = "urdf/franka_description/robots/franka_panda_gripper.urdf"

        # if "asset" in self.cfg["env"]:
        #     asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), self.cfg["env"]["asset"].get("assetRoot", asset_root))
        #     franka_asset_file = self.cfg["env"]["asset"].get("assetFileNameFranka", franka_asset_file)

        # load franka asset
        asset_options = gymapi.AssetOptions()
        asset_options.flip_visual_attachments = True
        asset_options.fix_base_link = True
        asset_options.collapse_fixed_joints = False
        asset_options.disable_gravity = True
        asset_options.thickness = 0.001
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_EFFORT
        asset_options.use_mesh_materials = True
        franka_asset = self.gym.load_asset(self.sim, asset_root, franka_asset_file, asset_options)

        if self.control_type == "osc":
            # If no franka passive damping 0, the arm will keep oscilating, does not know why. We use 10 if using impedance control
            arm_stiffness, arm_damping = 0, 0
            franka_dof_stiffness = to_torch([*[arm_stiffness]*7, 5000., 5000.], dtype=torch.float, device=self.device)
            franka_dof_damping = to_torch([*[arm_damping]*7, 1.0e2, 1.0e2], dtype=torch.float, device=self.device)
        else:
            arm_stiffness = 1000
            arm_damping = 2 * np.sqrt(arm_stiffness)
            franka_dof_stiffness = to_torch([*[arm_stiffness]*7, 5000., 5000.], dtype=torch.float, device=self.device)
            franka_dof_damping = to_torch([*[arm_damping]*7, 1.0e2, 1.0e2], dtype=torch.float, device=self.device)
        
        self.num_franka_bodies = self.gym.get_asset_rigid_body_count(franka_asset)
        self.num_franka_dofs = self.gym.get_asset_dof_count(franka_asset)
        print("num franka bodies: ", self.num_franka_bodies)
        print("num franka dofs: ", self.num_franka_dofs)

         # set franka dof properties
        franka_dof_props = self.gym.get_asset_dof_properties(franka_asset)
        self.franka_dof_lower_limits = []
        self.franka_dof_upper_limits = []
        self._franka_velocity_limits = []
        self._franka_effort_limits = []
        for i in range(self.num_franka_dofs):
            if self.control_type == "osc":
                franka_dof_props['driveMode'][i] = gymapi.DOF_MODE_POS if i > 6 else gymapi.DOF_MODE_EFFORT
            else:
                franka_dof_props['driveMode'][i] = gymapi.DOF_MODE_POS
            
            if self.physics_engine == gymapi.SIM_PHYSX:
                franka_dof_props['stiffness'][i] = franka_dof_stiffness[i]
                franka_dof_props['damping'][i] = franka_dof_damping[i]
            else:
                franka_dof_props['stiffness'][i] = 7000.0
                franka_dof_props['damping'][i] = 50.0

            # Constrain the gripper velocity to be 0.053 m/s (1.5s required for 0.08m open/close)
            # IsaacGym Bug: Setting the gripper velocity will affect other dofs instead of just the gripper!
            # franka_dof_props['velocity'][i] = 0.053 if i > 6 else franka_dof_props['velocity'][i]
            self.franka_dof_lower_limits.append(franka_dof_props['lower'][i])
            self.franka_dof_upper_limits.append(franka_dof_props['upper'][i])
            self._franka_velocity_limits.append(franka_dof_props['velocity'][i])
            self._franka_effort_limits.append(franka_dof_props['effort'][i])

        self.franka_dof_lower_limits = to_torch(self.franka_dof_lower_limits, device=self.device)
        self.franka_dof_upper_limits = to_torch(self.franka_dof_upper_limits, device=self.device)
        self._franka_velocity_limits = to_torch(self._franka_velocity_limits, device=self.device)
        self._real_franka_velocity_limits = self._franka_velocity_limits.clone()
        self._franka_effort_limits = to_torch(self._franka_effort_limits, device=self.device)
        # We manually set the effort limits to be 2 for the first 7 dofs to prevent the robot from moving too fast
        if self.cfg.get("torque_limits", None) is not None:
            self._franka_effort_limits[:7] = self.cfg.get("torque_limits")
            franka_dof_props['effort'][:7] = self.cfg.get("torque_limits")
        franka_dof_props['effort'][7:] = 80.0

        return franka_asset, franka_dof_props


    def custom_variable_init(self):
        self.dt = self.cfg.get("dt", 1/60)
        # Custom
        self.ratio_range = self.cfg.get("ratio_range", None)
        self.goal_speed = self.cfg.get("goal_speed", None)
        self.goal_time = self.cfg.get("goal_time", None)
        self.disturbance_v = self.cfg.get("disturbance_v", None)
        self.use_beta = self.cfg.get("beta", False)
        self.act_scale = self.cfg.get("act_scale", 1.)
        self.training = not self.cfg.get("eval_result", False)
        self.add_obs_noise = (self.training and self.cfg.get("add_obs_noise", False)) or self.cfg.get("apply_noise_eval", False)
        self.add_act_noise = (self.training and self.cfg.get("add_act_noise", False)) or self.cfg.get("apply_noise_eval", False)
  
        # Values to be filled in at runtime
        self.states = {}                        # will be dict filled with relevant states to use for reward calculation
        self.prev_states = {}                   # will be dict filled with previous states for reward calculation
        self.world_states = {}                  # all states in the world frame
        self.states_real = {}                   # Real world states
        self.prev_states_real = {}              # Previous real world states
        self.link_handles = {}                  # will be dict mapping names to relevant sim handles
        self.debug_info = {}
        self.num_dofs = None                    # Total number of DOFs per env
        self.actions = None                     # Current actions to be deployed
        self.focus_linvel_names = None           # Name of the object to focus linear velocity
        self.focus_linacc_names = None           # Name of the object to focus angular velocity

        # Tensor placeholders
        self._root_state = None             # State of root body        (n_envs, 13)
        self._contact_forces = None     # Contact forces in sim
        
        self._dof_state = None  # State of all joints       (n_envs, n_dof)
        self._q = None  # Joint positions           (n_envs, n_dof)
        self._qd = None                     # Joint velocities          (n_envs, n_dof)
        
        self._rigid_body_state = None  # State of all rigid bodies             (n_envs, n_bodies, 13)
        self._eef_state = None  # end effector state (at grasping point)
        self._eef_lf_state = None  # end effector state (at left fingertip)
        self._eef_rf_state = None  # end effector state (at left fingertip)

        self._j_eef = None  # Jacobian for end effector
        self._mm = None  # Mass matrix
        self._arm_control = None  # Tensor buffer for controlling arm
        self._arm_pos_control = None  # Position actions
        self._gripper_control = None  # Tensor buffer for controlling gripper
        self._gripper_mode = None  # Gripper mode (open/close)
        self._pos_control = None            # Position actions
        self._effort_control = None         # Torque actions
        self._franka_effort_limits = None        # Actuator effort limits for franka
        self._franka_velocity_limits = None      # Actuator velocity limits for franka
        self._real_franka_velocity_limits = None      # Actuator velocity limits for franka in the real world (slowly curriculum)
        self.franka_dof_lower_limits = None        # Actuator lower limits for franka
        self.franka_dof_upper_limits = None        # Actuator upper limits for franka
        self._global_indices = None         # Unique indices corresponding to all envs in flattened array

        # Env configuration
        self.max_episode_length = self.cfg.get("episodeLength", 800)
        self.start_position_noise = self.cfg["env"]["startPositionNoise"]
        self.start_rotation_noise = self.cfg["env"]["startRotationNoise"]
        self.franka_position_noise = self.cfg["env"]["frankaPositionNoise"]
        self.franka_rotation_noise = self.cfg["env"]["frankaRotationNoise"]
        self.franka_dof_noise = self.cfg["env"]["frankaDofNoise"]
        self.aggregate_mode = self.cfg["env"]["aggregateMode"]
        self.debug_vis = self.cfg.get("debug_vis", False)
        self.up_axis = "z"
        self.up_axis_idx = 2

        # Controller type
        self.control_type = self.cfg.get("control_type", "osc")
        assert self.control_type in {"osc", "ik", "jp"},\
            "Invalid control type specified. Must be one of: {osc, ik, jp}"
        self.control_freq_inv = self.cfg.get("control_freq_inv", 3)
        self.control_freq = 1. / self.dt / self.control_freq_inv
        self.gripper_freq_inv = self.cfg.get("gripper_freq_inv", 10)
        self.gripper_freq = self.control_freq // self.gripper_freq_inv
        assert self.control_freq % self.gripper_freq_inv == 0, \
            f"Gripper inv must be a multiple of control frequency, but got control: {self.control_freq} and gripper: {self.gripper_freq_inv}"
        assert self.gripper_freq <= 2, \
            f"Gripper frequency must be less than 2Hz, but got {self.gripper_freq}"
        self.ctrl_dt = self.dt * self.control_freq_inv
        
        self.domain_randomization_init()

    
    def domain_randomization_init(self):
        # Domain randomization settings
        self.dr_settings = {
            "spatial": {},
            "noise": {},
            "properties": {},
            "controller": {}
        }

        self.cur_dr_params = {
            k: {} for k in self.dr_settings.keys()
        }
        
        # Start to add the domain randomization settings
        self.dr_settings["spatial"].update({
            "franka_dof": [0., self.franka_dof_noise]
        })

        # Franka obs noise
        self.dr_settings["noise"].update({
            # We are using the absolute uniform noise for the observation; 
            # We are using the relative noise for the action and inertial
            # pos/dof in meters, quat in radians
            "eef_pos": [0., 0.005], # in all directions
            "eef_quat": [0., np.pi / 60], # 3 degrees
            "q_gripper": [0., 0.005], # 1cm in dof
            "q": [0., np.pi / 60], # 2 degrees

            # Action
            "action": [0., 0.1], # we do not apply noise to the action

            # Controller
            "inertial_mat": [0., 0.1], # 10% of the value (OSC only)
            "delta_Kp": [0, 25], # +- 25 | Absolute value
            "v_gripper": [0., 0.005], # +- 0.5cm/s | Absolute value
            "gripper_delay": [0., 0.05], # delay in seconds for gripper control
        })

        self.dr_settings["controller"].update({
            "max_vel_subtract": [0., self.cfg.get("max_vel_subtract", 0.)], # Gradually decrease to 0.2
            "alpha": [0., 0.9], # ema filter scaling
            "gripper_delay": [0.3, 0.3], # delay in seconds for gripper control
        })
        print("max_vel_subtract: ", self.cfg.get("max_vel_subtract", 0.))


    def common_data_init(self, env_ptr, franka_handle):
        franka_handle = self._franka_id
        table_handle = self._table_id
        # Franka handles for transformations
        self.franka_linkname2id_dict = self.gym.get_actor_rigid_body_dict(env_ptr, franka_handle)
        self.franka_linkid2name_dict = {v: k for k, v in self.franka_linkname2id_dict.items()}
        self.link_handles = {
            # Franka
            "hand": self.gym.find_actor_rigid_body_handle(env_ptr, franka_handle, "panda_hand"),
            "leftfinger_tip": self.gym.find_actor_rigid_body_handle(env_ptr, franka_handle, "panda_leftfinger_tip"),
            "rightfinger_tip": self.gym.find_actor_rigid_body_handle(env_ptr, franka_handle, "panda_rightfinger_tip"),
            "grip_site": self.gym.find_actor_rigid_body_handle(env_ptr, franka_handle, "panda_grip_site"),
            # Table
            "table": self.gym.get_actor_rigid_body_index(env_ptr, table_handle, 0, gymapi.DOMAIN_SIM),
        }

        # Get total DOFs and actor
        self.all_env_ids = torch.arange(self.num_envs, device=self.device)
        self.num_dofs = self.gym.get_sim_dof_count(self.sim) // self.num_envs
        self.num_actors = self.gym.get_sim_actor_count(self.sim) // self.num_envs
        self.num_bodies = self.gym.get_sim_rigid_body_count(self.sim) // self.num_envs

        # Setup tensor buffers
        _actor_root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        _dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        _rigid_body_state_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
        _contact_forces_tensor = self.gym.acquire_net_contact_force_tensor(self.sim)
        self._root_state = gymtorch.wrap_tensor(_actor_root_state_tensor).view(self.num_envs, -1, 13)
        self._dof_state = gymtorch.wrap_tensor(_dof_state_tensor).view(self.num_envs, -1, 2)
        self._rigid_body_state = gymtorch.wrap_tensor(_rigid_body_state_tensor).view(self.num_envs, -1, 13)
        self._contact_forces = gymtorch.wrap_tensor(_contact_forces_tensor).view(self.num_envs, -1, 3)
        self._q = self._dof_state[..., 0]
        self._qd = self._dof_state[..., 1]
        self._eef_state = self._rigid_body_state[:, self.link_handles["grip_site"], :]
        self._eef_lf_state = self._rigid_body_state[:, self.link_handles["leftfinger_tip"], :]
        self._eef_rf_state = self._rigid_body_state[:, self.link_handles["rightfinger_tip"], :]
        self._table_contact_forces = self._contact_forces[:, self.link_handles["table"], :]
        self._arm_contact_forces = self._contact_forces[:, :self.gym.get_actor_rigid_body_count(env_ptr, franka_handle), :]
        _jacobian = self.gym.acquire_jacobian_tensor(self.sim, "franka")
        jacobian = gymtorch.wrap_tensor(_jacobian)
        hand_joint_index = self.gym.get_actor_joint_dict(env_ptr, franka_handle)['panda_hand_joint']
        self._j_eef = jacobian[:, hand_joint_index, :, :7]
        _massmatrix = self.gym.acquire_mass_matrix_tensor(self.sim, "franka")
        mm = gymtorch.wrap_tensor(_massmatrix)
        self._mm = mm[:, :7, :7]

        # Initialize control buffers
        self._pos_control = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float, device=self.device)
        self._effort_control = torch.zeros_like(self._pos_control)

        # Initialize control
        self._arm_control = self._effort_control[:, :7]
        self._arm_pos_control = self._pos_control[:, :7]
        self._gripper_control = self._pos_control[:, 7:9]
        self._gripper_mode = -torch.ones(self.num_envs, dtype=torch.float, device=self.device)
        self._gripper_mode_temp = -torch.ones(self.num_envs, dtype=torch.float, device=self.device)  # Temporary gripper mode for control
        self.gripper_ctrl_counts = torch.zeros(self.num_envs, device=self.device)
        self.gripper_delay_timer = torch.zeros(self.num_envs, device=self.device)
        self.gripper_ctrl_counts_real = 0

        # Initialize indices
        self._global_indices = torch.arange(self.num_envs * self.num_actors, dtype=torch.int32,
                                            device=self.device).view(self.num_envs, -1)
        
        # Work space limits
        self._ws_surface_pos = to_torch(self._ws_surface_pos, device=self.device)
        self._ws_upper_bounds = to_torch([0.2] * 2, device=self.device)
        self.MAX_CONTACT_FORCE_NORM = 20.
        self.MAX_VEL_NORM = 10.


    def obs_act_rew_init(self):
        # Add custom observations
        self.cfg["env"]["numObservations"] += 2 # time_ratio (1) + time_to_end (1)
        
        # actions include: delta EEF if OSC (6) or joint torques (7) + bool gripper (1)
        if self.control_type in ["osc", "ik"]:
            self.cfg["env"]["numActions"] = 7
        elif self.control_type == "jp":
            self.cfg["env"]["numActions"] = 8

        self.cfg["env"]["numStates"] = self.cfg["env"]["numObservations"]
        self.prim_obs_init()
        
        # Memory Buffer Length
        self.numSingleObs = self.cfg["env"]["numObservations"]
        self.numSingleState = self.cfg["env"]["numStates"]
        self.sequence_len = self.cfg.get("sequence_len", 1)
        self.cfg["env"]["numObservations"] *= self.sequence_len
        self.cfg["env"]["numStates"] *= self.sequence_len
        
        # Add custom rewards
        task_dense_rewards = self.reward_settings.copy()
        self.reward_settings.update({
            "r_success":self.cfg.get("successRewardScale", 1000.),
            "r_violate": self.cfg.get("violateRewardScale", 0.),
            "r_action_penalty_scale": self.cfg["env"].get("actionPenaltyScale", 0.),
            "r_force_penalty_scale": self.cfg["env"].get("forcePenaltyScale", 0.),
            # Continuous Checking Reward Special
            "r_hold_scale": self.cfg["env"].get("holdRewardScale", 0.),

            # Time Aware
            "r_epstime_scale": self.cfg.get("epstimeRewardScale", [0.])[0],
            "r_scene_vel_scale": self.cfg.get("scevelRewardScale", [0.])[0],
            "r_steptime_scale": self.cfg.get("steptimeRewardScale", 0.),
            "r_scene_acc_scale": self.cfg.get("sceaccRewardScale", 0.),
            "r_arm_vel_scale": self.cfg.get("actvelPenaltyScale", 0.),
            "r_arm_acc_scale": self.cfg.get("actaccPenaltyScale", 0.),
        })

        if self.cfg.get("no_dense", False):
            for key in task_dense_rewards:
                self.reward_settings[key] = 0.

    
    def prim_obs_init(self):
        self.cfg["env"]["numStates"] += 2 # sce_linvel (1) + max_linvel_gt (1)
        self.task_prim_obs_init()

    
    def timeaware_init(self):
        # Specify the idxs for custom observations
        t2e_idx = -1
        self.extras["tobs_idx"] = t2e_idx
        self.extras["vobs_idx"] = -3

        self.franka_related()
        self.configs_related()
        self.argument_related()
        self.maskout_related()
        self.stage_wise_ctrl_related()


    def franka_related(self):
        # Franka defaults
        self.use_real_pure = self.cfg.get("real_robot", False) and not self.cfg.get("use_sim_pure", False)
        self.franka_default_dof_pos = to_torch(
            [0, 0, 0, -2.3180, 0, 2.4416, 0.7854, 0.04, 0.04], device=self.device
        )

        # Number of Interpolate Joints for each control command 
        self.num_inter_steps = self.cfg.get("interpolate_joints", 1)
        assert self.num_inter_steps >= 1, "Number of interpolation steps must be at least 1."
        if self.num_inter_steps > 1:
            if self.training:
                # Because the control dt is incorrect in this case
                raise ValueError("Interpolation is not supported during training. Please set 'interpolate_joints' to 1.")
            if self.control_type == "osc":
                raise ValueError("Interpolation is not supported for OSC control type. Please set 'interpolate_joints' to 1.")
        self.max_episode_length *= self.num_inter_steps

        # Gains in the simulation (osc)
        self.kp_init = 150
        self.kp = torch.ones((self.num_envs, 6), device=self.device) * self.kp_init
        self.kd = 2 * torch.sqrt(self.kp)
        self.kp_null = torch.ones((self.num_envs, 7), device=self.device) * (self.kp_init / 15)
        self.kd_null = 2 * torch.sqrt(self.kp_null)

        # Gains in the real world (osc)
        self.kp_real_init = 100
        self.kp_real = torch.ones((1, 6), device=self.device) * self.kp_real_init
        self.kd_real = 2 * torch.sqrt(self.kp_real)
        self.kp_null_real = torch.ones((1, 7), device=self.device) * (self.kp_real_init / 15)
        self.kd_null_real = 2 * torch.sqrt(self.kp_null_real)

        # Gains joint position pd control (jp)
        self.kp_jp_init = 100
        self.kp_jp = torch.ones((self.num_envs, 7), device=self.device) * self.kp_jp_init
        self.kd_jp = 2 * torch.sqrt(self.kp_jp)

        # Set control limits
        if self.control_type == "osc":
            self.cmd_limit = to_torch([0.1, 0.1, 0.1, 0.5, 0.5, 0.5], device=self.device).unsqueeze(0) / self.cfg.get("control_freq_inv", 1)
        elif self.control_type == "ik":
            self.cmd_limit = to_torch([0.1, 0.1, 0.1, 0.5, 0.5, 0.5], device=self.device).unsqueeze(0) / self.cfg.get("control_freq_inv", 1)
        else:
            self.cmd_limit = to_torch([1.]*7, device=self.device).unsqueeze(0) / self.cfg.get("control_freq_inv", 1)

        # Use for transformation
        self.unit_z = torch.tensor([0., 0., 1.], device=self.device).repeat(self.num_envs, 1)
        self.unit_quat = torch.tensor([0., 0., 0., 1.], device=self.device).repeat(self.num_envs, 1)

    
    def change_controller_gains(self, env_ids):
        # OSC Gains; Original Kp is 150 and Kd is 10
        delta_kp = torch_rand_float(-self.cur_dr_params["noise"]["delta_Kp"], self.cur_dr_params["noise"]["delta_Kp"], (len(env_ids), 1), device=self.device)
        
        self.kp[env_ids] = self.kp_init + delta_kp.repeat(1, 6)
        self.kd[env_ids] = 2 * torch.sqrt(self.kp[env_ids])
        self.kp_null[env_ids] = (self.kp[env_ids, 0]/15).unsqueeze(-1).repeat(1, 7)
        self.kd_null[env_ids] = 2 * torch.sqrt(self.kp_null[env_ids])


    def update_max_velocity(self):
        self._real_franka_velocity_limits[:7] = self._franka_velocity_limits[:7] * (1 - self.cur_dr_params["controller"]["max_vel_subtract"])
        if self.cfg.get("limit_gripper_vel", True):
            self._real_franka_velocity_limits[7:] = self._franka_velocity_limits[7:] * (1 - self.cur_dr_params["controller"]["max_vel_subtract"])
            self._real_franka_velocity_limits[7:] = torch.clamp(self._real_franka_velocity_limits[7:], min=0.05) # minimum 0.05 m/s gripper


    def configs_related(self):
        # Read initial configuration
        if self.cfg.get("fixed_configs", False):
            if self.cfg.get("global_configs", False):
                json_file = os.path.join("eval_res", self.cfg["task_name"], self.cfg["global_ckpt"]+f"_EVAL_{self.cfg['global_index_episode']}", "trajectories", "init_configs.json")
            elif self.cfg.get("par_configs", False): # update previous configs
                json_file = os.path.join("eval_res", self.cfg["task_name"], self.cfg["par_checkpoint"]+f"_EVAL_{self.cfg['par_index_episode']}", "trajectories", "init_configs.json")
            else:
                json_file = os.path.join("eval_res", self.cfg["task_name"], self.cfg["checkpoint"]+f"_EVAL_{self.cfg['index_episode']}", "trajectories", "init_configs.json")
            
            assert os.path.exists(json_file), f"Initial configuration file not found: {json_file}"
            with open(json_file, "r") as f:
                self.env_configs = json.load(f)
            # convert to torch
            for key in self.env_configs:
                self.env_configs[key] = torch.tensor(self.env_configs[key], device=self.device)
            self.num_configs = len(self.env_configs["time_used"])

            if self.cfg.get("update_configs", False):
                assert self.num_envs >= self.num_configs, f"Number of environments ({self.num_envs}) must be larger than the number of initial configurations for parallel updating ({self.num_configs})."
                self.config_ids = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
                self.config_ids[:self.num_configs] = torch.arange(self.num_configs, device=self.device, dtype=torch.long)
                if self.cfg.get("target_success_eps", 0) != self.num_configs:
                    print(f"**************\n"
                          f"WARNING: target_success_eps ({self.cfg['target_success_eps']}) is indifferent with the number of initial configurations ({self.num_configs})."
                          f"We will only use the number of initial configurations ({self.num_configs}).\n"
                          f"**************")
                    
            # Episode time check
            if self.ratio_range is not None:
                req_max_len = (self.env_configs["time_used"].max().item()) / self.ratio_range[0] / self.dt
                cfg_max_len = self.max_episode_length
                assert (req_max_len+300) <= cfg_max_len, f"The maximum episode length ({cfg_max_len}) is smaller than Maximum episode length needed ({req_max_len+300}). Please increase to ({np.ceil(req_max_len)+300})."

            # Compute the average time used and max linvel
            self.avg_t2e = 0.
            self.avg_limvel = 0.
            if self.cfg.get("use_avg_t2e", False):
                self.avg_t2e = self.env_configs["time_used"][:] = self.env_configs["time_used"].mean()
            if self.cfg.get("use_avg_limvel", False):
                self.avg_limvel = self.env_configs["max_linvel"][:] = self.env_configs["max_linvel"].mean()
            elif self.cfg.get("use_max_limvel", False):
                self.avg_limvel = self.env_configs["max_linvel"][:] = self.env_configs["max_linvel"].max()

            if self.cfg.get("max_dist", False):
                self.cfg["specific_idx"] = torch.argmax(self.env_configs["time_used"]).item()


    def get_config_idx(self, env_ids):
        config_idx = None
        if self.cfg.get("fixed_configs", False):
            if self.cfg.get("specific_idx", None) is not None:
                config_idx = torch.tensor([self.cfg["specific_idx"]], device=self.device, dtype=torch.long)
            elif self.cfg.get("update_configs", False):
                config_idx = self.config_ids[env_ids]
            else:
                config_idx = torch.randint(0, self.num_configs, (len(env_ids),), device=self.device)

        return config_idx
    

    def filter_env_ids(self, env_ids, config_index):
        # filter the env_ids that not reach the save_threshold
        if self.is_ready_to_record(env_ids):
            # Some envs might keep failing, we need to reset them
            failed_env_ids = (self.reset_buf * (1 - self.success_buf)).nonzero(as_tuple=False).squeeze(-1)
            self.failure_counts[failed_env_ids] += 1
            failed_too_many_idx = self.failure_counts[env_ids] >= self.save_threshold
            record_done_idx = self.success_counts[env_ids] >= self.save_threshold
            
            new_cfg_sub_ids = (record_done_idx | failed_too_many_idx).nonzero(as_tuple=False).squeeze(-1)
            new_cfg_env_ids = env_ids[new_cfg_sub_ids]
            # Update the config_index if in the update_configs mode
            config_index = config_index[new_cfg_sub_ids] if config_index is not None else None
            self.success_counts[new_cfg_env_ids] = 0
            self.failure_counts[new_cfg_env_ids] = 0
        else:
            new_cfg_env_ids = env_ids
        
        return new_cfg_env_ids, config_index
            
                    
    def argument_related(self):
        self.save_threshold = self.cfg.get("save_threshold", 10) # 100 successful trials for one configuration to save such configuration
        self.success_counts = torch.ones(self.num_envs, device=self.device) * self.save_threshold # reset all envs at the beginning
        self.failure_counts = torch.zeros(self.num_envs, device=self.device)
        self.num_episodes = 0 # count the number of episodes
        self.saved_eps = [] # count the number of episodes recorded
        self.time_used_accm = []
        self.max_linvel_accm = []
        self.sum_linvel_accm = []
        self.extras["franka"] = {
            "dof_lower": self.franka_dof_lower_limits.cpu().numpy(),
            "dof_upper": self.franka_dof_upper_limits.cpu().numpy(),
            "velocity_limits": self._franka_velocity_limits.cpu().numpy(),
            "effort_limits": self._franka_effort_limits.cpu().numpy(),
        }

        self.max_eps_time = self.max_episode_length * self.dt
        self.obs_range = {
            "time_to_end": torch.tensor([0., self.max_eps_time], device=self.device),
            "time_ratio": torch.tensor([0.2, 1.], device=self.device),
            "max_linvel": torch.tensor([0., 1.5], device=self.device),
        }


    def maskout_related(self):
        # Mask out the gripper actions if the gripper is not used
        self.maskout_names_full = ["time_ratio", "time_to_end", "sce_linvel", "lim_linvel"]
        self.maskout_names = []
        
        maskout_names = []
        if self.cfg.get("pre_train", False) or self.cfg.get("fix_priv", False):
            maskout_names = self.maskout_names_full.copy()
        if not self.cfg.get("time2end", False):
            maskout_names += ["time_ratio", "time_to_end"]
        if self.cfg.get("fix_linvel", False):
            maskout_names += ["sce_linvel"]
        if self.cfg.get("fix_limvel", False):
            maskout_names += ["lim_linvel"]

        for name in maskout_names:
            if name not in self.maskout_names:
                self.maskout_names.append(name)


    def stage_wise_ctrl_related(self):
        self.use_staged_ctrl = True if self.cfg.get("budget_portion", None) is not None else False
        if self.use_staged_ctrl:
            self.budget_portion = budget_portion = self.cfg["budget_portion"]
            self.speed_describe = speed_describe = self.cfg["speed_describe"]
            self.fast_portion = self.slow_portion = 0.
            for i in range(len(budget_portion)):
                if speed_describe[i] == 1:
                    self.fast_portion += budget_portion[i]
                else:
                    self.slow_portion += budget_portion[i]

            assert np.allclose(sum(self.budget_portion), 1), f"Budget portion must sum to 1, but get {sum(self.budget_portion)}."
            assert (self.fast_portion > 0) and (self.slow_portion > 0), "Budget portion must have both fast and slow portions."
            self.stage_time_ratio_buf = torch.zeros((self.num_envs, len(budget_portion)), device=self.device)
            self.real_time_milestone = torch.tensor([sum(self.budget_portion[:i+1]) * self.goal_time for i in range(len(budget_portion))]).repeat(self.num_envs, 1).to(self.device)
            self.speed_describe_tensor = torch.tensor(speed_describe, device=self.device).repeat(self.num_envs, 1)
            self.cur_stage = torch.zeros((self.num_envs, ), device=self.device, dtype=torch.long)


    def _warmup_env(self):
        # Reset all environments twice (isaacgym bug; the set_dof_position_target_tensor_indexed at the beginning needs to run twice for properly work)
        for _ in range(2):
            self.reset_idx(self.all_env_ids)

        # Refresh tensors using few steps of simulation
        for i in range(10):
            self.gym.simulate(self.sim)
        self._refresh()
        self._pre_compute_tf()
        # Compute the initial observations
        self.compute_observations(reset_ids=torch.arange(self.num_envs, device=self.device))
    
    
    def _pre_compute_tf(self):
        self._franka2W_quat, self._franka2W_pos = tf_inverse(self._franka_state[:, 3:7], self._franka_state[:, :3])

    
    def _update_task_prev_states(self):
        """ Will be filled by each task script"""
        pass
    
    
    def _update_prev_states(self, real=False):
        """Update the previous states of the environment. This is called at the beginning of each step."""
        # Update common prev states
        states, prev_states = self.get_states_dict(real=real)
        for name in ["q", "q_vel", "q_gripper", "q_gripper_vel"]:
            if name not in states: continue
            prev_states[name] = mix_clone(states[name])
        self._update_task_prev_states(real=real)
    
    
    def _update_task_states(self):
        """ Will be filled by each task script"""
        pass


    def _update_robot_states(self):
        # convert transformations from the world frame to the robot base frame
        eef_quat, eef_pos = self.point2frankaBase(self._eef_state[:, 3:7], self._eef_state[:, :3])
        _, eef_lf_pos = self.point2frankaBase(self._eef_lf_state[:, 3:7], self._eef_lf_state[:, :3])
        _, eef_rf_pos = self.point2frankaBase(self._eef_rf_state[:, 3:7], self._eef_rf_state[:, :3])
        _, eef_unitz_pos = tf_combine(eef_quat, eef_pos, self.unit_quat, self.unit_z)
        
        # We assume the franka base orientation is the same as the world frame; so we do not convert the velocity
        # eef_linvel = self.vec2frankaBase(self._eef_state[:, 7:10])
        # eef_angvel = self.vec2frankaBase(self._eef_state[:, 10:])
        # eef_vel = torch.cat([eef_linvel, eef_angvel], dim=-1)
        eef_vel = self._eef_state[:, 7:]

        self.states.update({
            # Franka
            "q": self._q[:, :self.num_franka_dofs-2],
            "q_gripper": self._q[:, self.num_franka_dofs-2:self.num_franka_dofs],
            "q_vel": self._qd[:, :self.num_franka_dofs-2],
            "q_gripper_vel": self._qd[:, self.num_franka_dofs-2:self.num_franka_dofs],
            "prev_tgtq": self.prev_tgtq,
            "prev_dq": self.prev_dq,
            "eef_pos": eef_pos,
            "eef_quat": eef_quat,
            "eef_vel": eef_vel,
            "eef_lf_pos": eef_lf_pos,
            "eef_rf_pos": eef_rf_pos,
            "eef_unitz_pos": eef_unitz_pos,
            "j_eef": self._j_eef,
            # Force
            "table_contact_forces": self._table_contact_forces,
            "arm_contact_forces": self._arm_contact_forces,
            # Controller
            "gripper_mode": self._gripper_mode_temp.view(-1, 1),
            "gripper_waiting": self.gripper_ctrl_counts.view(-1, 1)
        })

        self.world_states.update({
            # Franka World Frame
            "eef_pos": self._eef_state[:, :3],
            "eef_quat": self._eef_state[:, 3:7],
            "eef_vel": self._eef_state[:, 7:],
            "eef_lf_pos": self._eef_lf_state[:, :3],
            "eef_rf_pos": self._eef_rf_state[:, :3],
        })


    def _update_stage_wise_time_ratio(self):
        if self.use_staged_ctrl:
            all_env_ids = torch.arange(self.cur_stage.size(0), device=self.device)
            to_next_stage_env_ids = torch.where(self.time_cur_buf > self.real_time_milestone[all_env_ids, self.cur_stage])[0]
            self.cur_stage[to_next_stage_env_ids] = torch.clamp(self.cur_stage[to_next_stage_env_ids] + 1, max=len(self.budget_portion) - 1)
            self.update_time_ratio_buf(self.stage_time_ratio_buf[all_env_ids, self.cur_stage], env_ids=all_env_ids)
            self.update_linvel_gt()


    def _update_time_to_end(self, running_env_index):
        if self.cfg.get("time_ratio", False):
            step_dt = self.ctrl_dt * self.time_ratio_buf
            self.time_to_end[running_env_index] -= step_dt[running_env_index]
        else:
            step_dt = self.ctrl_dt
            self.time_to_end[running_env_index] = self.time_to_end[running_env_index] * (self.prev_time_ratio_buf[running_env_index] / self.time_ratio_buf[running_env_index]) - step_dt

        if not self.training and self.cfg.get("real_robot", False):
            print(f"CurTime2End: {self.time_to_end[0].item():.5f}; TimeBudgetReal: {self.real_time_to_end[0].item():.5f}")


    def _update_scene_instability(self, real=False):
        """
        Update the scene instability based on the current linear velocity of the focused object.
        This is used to compute the scene instability penalty.
        """
        states, prev_states = self.get_states_dict(real=real)
        self.sce_linvel_buf[:] = torch.vstack([to_torch(states[name]).flatten() for name in self.focus_linvel_names]).sum(dim=0)
        self.max_linvel_buf = torch.where(self.max_linvel_buf < self.sce_linvel_buf, self.sce_linvel_buf, self.max_linvel_buf)
    

    def _update_manipulation_time(self, running_env_index):
        running_env_ids = running_env_index.nonzero(as_tuple=False).flatten()
        if len(running_env_ids) > 0:
            manipulating_index = self.sce_linvel_buf[running_env_index] >= 0.01
            manipulation_ids = running_env_ids[manipulating_index]
            if len(manipulation_ids) > 0:
                # Update the manipulation time for the running environments
                self.interaction_time[manipulation_ids] += self.ctrl_dt

    
    def _update_timeaware_states(self, real=False):
        # When the env is reset, the root/dof state can be imediately updated if using "set". 
        # However the eef pose will not be updated since it is not directly set, introducing idle states.
        # The best way is to use extra step simulation after the reset.
        states, prev_states = self.get_states_dict(real=real)
        running_env_index = self.continuous_check_buf==0 # We do not update any time during the continuous checking
        self.time_cur_buf[running_env_index] += self.ctrl_dt
        self.real_time_to_end[running_env_index] -= self.ctrl_dt
        self._update_stage_wise_time_ratio()
        self._update_time_to_end(running_env_index)
        self._update_scene_instability(real=real)
        self._update_manipulation_time(running_env_index)

        timeaware_states = {
            # Time
            "time_cur": self.time_cur_buf,
            "time_ratio": self.time_ratio_buf,
            "time_to_end": self.time_to_end,
            # Velocity
            "sce_linvel": self.sce_linvel_buf,
            "max_linvel": self.max_linvel_buf,
            "lim_linvel": self.linvel_max_gt,
            # Acc (All 0 for now; did not used and update)
            "sce_linacc": self.sce_linacc_buf
        }
        
        # Convert to numpy if real
        for key, value in timeaware_states.items():
            if real:
                timeaware_states[key] = value[0].cpu().numpy()
            else:
                timeaware_states[key] = value.view(-1, 1) if value.dim() == 1 else value
        
        states.update(timeaware_states)
        
    
    def _reset_prev_states(self, reset_ids):
        if len(reset_ids) > 0: 
        # Reset previous states to states
            remove_keys = []
            for key in self.prev_states:
                if not key in self.states:
                    remove_keys.append(key)
                else:
                    self.prev_states[key][reset_ids] = self.states[key][reset_ids].clone()
            
            for rm_key in remove_keys:
                self.prev_states.pop(rm_key)


    def _update_diff_states(self, real=False):
        """ Will be filled in the child envs"""
        pass

    
    def _update_states(self, reset_ids=[]):
        """
        Will be filled in the child envs
        """
        self._update_task_states()
        self._update_robot_states()
        # manually reset the previous states here; because issacgym need one more step to refresh the states
        self._reset_prev_states(reset_ids)
        self._update_diff_states()
        self._update_timeaware_states()
        self._update_prev_states()
        self._unify_quat_states()


    def _update_common_info(self, reward_records):
        self.extras.update({
            "eps_time": self.time_cur_buf,
            "eps_horizon": self.progress_buf,
            "eps_time_p": reward_records.get("eps_time_p", torch.zeros_like(self.rew_buf)),
            "scene_linvel_penalty": reward_records.get("scene_linvel_penalty", torch.zeros_like(self.rew_buf)),
            "scene_linacc_penalty": reward_records.get("scene_linacc_penalty", torch.zeros_like(self.rew_buf)),
            "eps_lim_scevel": self.states["lim_linvel"].flatten(),
            "eps_max_scevel": self.states["max_linvel"].flatten(),
            "eps_sum_inst": reward_records["accm_instability"] if "accm_instability" in reward_records else torch.zeros_like(self.rew_buf),
            "arm_qvel_penalty": reward_records["arm_qvel_penalty"] if self.reward_settings["r_arm_vel_scale"]!=0 else torch.zeros_like(self.rew_buf),
            "interaction_time": self.interaction_time.flatten(),
        })

        if not self.training:
            gripper_vel = self.states["q_gripper_vel"][0].clone()
            gripper_v_fix = (self.states["q_gripper"][0] - self.prev_states["q_gripper"][0]).clone() / self.ctrl_dt # gripper vel does not make sense in pos control mode. We use the difference of pos to get the vel
            gripper_vel[:] = gripper_v_fix
            self.extras.update({
                # Time related
                "observed_time2end": self.time_to_end[0].clone(),
                "real_time2end": self.real_time_to_end[0].clone(),
                "real_cur_time": self.time_cur_buf[0].clone(),
                "time_ratio": self.time_ratio_buf[0].clone(),
                "eps_time_goal": self.real_time2end_init.clone(),
                # Scene related
                "scene_linvel": self.states["sce_linvel"][0].clone(),
                "scene_linvel_lim": self.states["lim_linvel"][0].clone(), # this is used for visualization
                # Franka Related
                "joint_tgt_q": self._arm_pos_control[0].clone(),
                "joint_torqs": self._effort_control[0].clone(),
                "joint_poss": self.states["q"][0].clone(),
                "joint_vels": self.states["q_vel"][0].clone(),
                "joint_accs": (self.states["q_vel"][0] - self.prev_states["q_vel"][0]) / self.ctrl_dt, # refer torque is better
                "joint_gripper_poss": self.states["q_gripper"][0].clone(),
                "joint_gripper_vels": gripper_vel,
                "joint_velocity_limits": self._real_franka_velocity_limits.clone(),
            })

            if self.force_render:
                self.extras.update({
                    # Sim Image
                    "env_image": self.get_viewer_image(env_id=0)
                })

            if self.cfg.get("use_fk_replay", False):
                joint_tgt_q = self.extras["joint_tgt_q"].tolist()
                gripper_mode = self._gripper_mode_temp[0].cpu().item()
                u_gripper = 0 if gripper_mode==self.prev_u_gripper_real else gripper_mode
                self.prev_u_gripper_real = gripper_mode
                ctrl_cmd = [*joint_tgt_q, u_gripper]
                self.extras["fk_replay_cmd"] = ctrl_cmd


    def _reset_bufs(self, env_ids):
        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0
        self.continuous_check_buf[env_ids] = 0
        self.force_has_applied[env_ids] = 0
        self._gripper_mode[env_ids] = -1
        self._gripper_mode_temp[env_ids] = -1
        # Reset previous buf
        for i in range(self.sequence_len):
            self.state_queue[i][env_ids] = 0.
            self.obs_queue[i][env_ids] = 0.
        self.prev_state[env_ids] = 0.
        self.prev_obs[env_ids] = 0.
        self.prev_tgtq[env_ids] = self.fk_init_dof[env_ids, :7]
        self.prev_tgtq_gripper[env_ids] = self.fk_init_dof[env_ids, 7:9]
        self.prev_dq[env_ids] = 0.
        self.prev_u_gripper_real = -1
        self._reset_timeaware_bufs(env_ids)
    
    
    def _reset_timeaware_bufs(self, env_ids):
        # Reset Timer and max linvel are used in the observation.
        self.time_cur_buf[env_ids] = 0.
        self.interaction_time[env_ids] = 0.
        self.sce_linvel_buf[env_ids] = 0.
        self.max_linvel_buf[env_ids] = 0.
        self.sce_linacc_buf[env_ids] = 0.
        self.accm_instability[env_ids] = 0.
        self.time_to_end[env_ids] = self.time2end_init[env_ids]
        self.real_time_to_end[env_ids] = self.real_time2end_init[env_ids]
        self.prev_time_ratio_buf[env_ids] = self.time_ratio_buf[env_ids].clone()
        self.recompute_staged_time_ratio(env_ids)


    def compute_observations(self, reset_ids=[]):
        """Compute the observations for the environment.

        Args:
            reset_ids: the indices of the environments to reset.
        """
        # Refresh states
        self._update_states(reset_ids)
        self._collect_process_obs()
        self._collect_process_states()

        
    def _collect_process_obs(self):
        task_obs_names = self.get_taskobs_names()
        obs_names = self.add_timeaware_obs(task_obs_names)
        cur_obs = self._stacking_obs(obs_names, self.add_obs_noise)
        cur_fix_obs = cur_obs.clone()
        cur_obs = self.maskout_buf(cur_obs, self.obs_slice)
        self.obs_buf[:] = self.update_memory_buf(self.obs_queue, cur_obs)
        self.prev_obs[:] = cur_obs
        if self.cfg.get("stu_train", False):
            self.fixed_obs_buf[:] = self.maskout_all_timeaware(cur_fix_obs, self.obs_slice)


    def _collect_process_states(self):
        task_state_names = self.get_taskobs_names()
        task_state_names = self.add_priv_taskobs(task_state_names)
        state_names = self.add_timeaware_obs(task_state_names)
        state_names = self.add_priv_timeaware_obs(task_state_names)
        cur_state = self._stacking_states(state_names)
        cur_fix_state = cur_state.clone()
        cur_state = self.maskout_buf(cur_state, self.state_slice)
        self.states_buf[:] = self.update_memory_buf(self.state_queue, cur_state)
        self.prev_state[:] = cur_state
        if self.cfg.get("stu_train", False):
            self.fixed_state_buf[:] = self.maskout_all_timeaware(cur_fix_state, self.state_slice)

    
    def update_observations_dict(self):
        # Fixed observation is similar to observation but some time-related value are fixed
        self.obs_dict["states"] = torch.clamp(self.states_buf, -self.clip_obs, self.clip_obs).to(self.rl_device)
        self.obs_dict["obs"] = torch.clamp(self.obs_buf, -self.clip_obs, self.clip_obs).to(self.rl_device)
        if self.cfg.get("stu_train", False):
            self.obs_dict["fixed_state"] = torch.clamp(self.fixed_state_buf, -self.clip_obs, self.clip_obs).to(self.rl_device)
            self.obs_dict["fixed_obs"] = torch.clamp(self.fixed_obs_buf, -self.clip_obs, self.clip_obs).to(self.rl_device)
    
    
    def compute_reward(self, actions):
        rewards, reset_buf, success_buf = self.compute_task_reward()
        rewards, reward_records = self.apply_timeaware_rewards(rewards, reset_buf, success_buf)
        self.rew_buf[:], self.reset_buf[:], self.success_buf[:] = rewards, reset_buf, success_buf

        # Update episodes count
        self.num_episodes += self.reset_buf.sum().item()
        # Update info
        self._update_common_info(reward_records)
    
    
    def apply_timeaware_rewards(self, rewards, reset_buf, success_buf):
        cfg = self.cfg
        reward_settings = self.reward_settings
        states = self.states
        prev_states = self.prev_states
        accm_instability = self.accm_instability

        real_time_to_end = self.real_time_to_end
        cost = torch.zeros(self.num_envs, self.cfg.get("num_cost", 2), device=self.device)
        
        # Timeaware rewards
        reward_records = {}
        if reward_settings["r_epstime_scale"] > 0:
            eps_time_cost = torch.zeros_like(real_time_to_end)
            eps_time_reward = torch.zeros_like(real_time_to_end)
            success_flag = success_buf == 1
            failed_flag = (reset_buf == 1) & (success_buf == 0)
            if self.ratio_range is None:
                # Maximize time2end to become faster
                eps_time_reward[success_flag] = real_time_to_end[success_flag]
            else:
                # Minimize time2end to be on time (time aware FT); We weight the time cost by the speed ratio
                eps_time_cost[success_flag] = torch.abs(real_time_to_end[success_flag])
                eps_time_reward = -torch.clamp(eps_time_cost, max=5.) # Maximum time mismatch is 5 seconds

            rewards += reward_settings["r_epstime_scale"] * eps_time_reward
            rewards = torch.clamp(rewards, min=0.) # Do not allow negative rewards
            reward_records["eps_time_p"] = eps_time_cost

        # Stability rewards
        if self.ratio_range is not None:
            arm_qvel = states["q_vel"]
            arm_qacc = (arm_qvel - prev_states["q_vel"]) / self.ctrl_dt
            arm_qvel_penalty = torch.norm(arm_qvel, dim=-1)
            arm_qacc_penalty = torch.norm(arm_qacc, dim=-1)
            arm_qvel_reward = -arm_qvel_penalty
            arm_qacc_reward = -arm_qacc_penalty

            focus_linvel_norm = self.states["sce_linvel"].flatten()
            focus_linacc_norm = self.states["sce_linacc"].flatten()
            if cfg.get("vel_match", False): # change scene_linvel_sum to scene_linvel_sum
                scene_linvel_sum = torch.clamp(focus_linvel_norm-self.linvel_max_gt, min=0.)
            else:
                scene_linvel_sum = focus_linvel_norm
            scene_linvel_sum = torch.clamp(scene_linvel_sum, max=self.MAX_VEL_NORM)

            scene_linvel_reward = -scene_linvel_sum
            scene_linacc_reward = -focus_linacc_norm
            cost[:, 1] = scene_linvel_sum

            # Stage control we only record the stabe stage
            recording_env_ids = self.all_env_ids if not self.use_staged_ctrl else \
                torch.where(self.speed_describe_tensor[self.all_env_ids, self.cur_stage] == 0)[0]
            accm_instability[recording_env_ids] += focus_linvel_norm[recording_env_ids]
            accm_instability_reward = -accm_instability
            if not self.cfg.get("use_cost", False):
                rewards += (reward_settings["r_scene_vel_scale"] * scene_linvel_reward + reward_settings["r_arm_vel_scale"] * arm_qvel_reward)
                rewards += (reward_settings["r_scene_acc_scale"] * scene_linacc_reward + reward_settings["r_arm_acc_scale"] * arm_qacc_reward)
            reward_records["accm_instability"] = accm_instability
            reward_records["scene_linvel_penalty"] = scene_linvel_sum
            reward_records["scene_linacc_penalty"] = focus_linacc_norm
            reward_records["scene_linvel"] = focus_linvel_norm
            reward_records["arm_qvel_penalty"] = arm_qvel_penalty
            reward_records["q_acc"] = arm_qacc

        self.extras.update({
            "cost": cost,
        })
        
        return rewards, reward_records
    

    def add_timeaware_obs(self, obs_names):
        obs_names += ["time_ratio", "time_to_end"]
        return obs_names
    

    def add_priv_timeaware_obs(self, state_names):
        state_names += ["sce_linvel", "lim_linvel"]
        return state_names
    

    def _stacking_obs(self, obs_names, add_obs_noise=False):
        obs = []
        index = 0
        for name in obs_names:
            post_obs_piece = obs_piece = self.states[name]
            len_obs = obs_piece.shape[-1]
            if add_obs_noise and name in self.cur_dr_params["noise"].keys():
                noise_scale = self.cur_dr_params["noise"].get(name, 0.0)
                if "quat" in name:
                    noise_v = quat_from_euler_xyz(*torch_rand_float(-noise_scale, noise_scale, (3, self.num_envs), device=self.device))
                    post_obs_piece = normalize(quat_mul(obs_piece, noise_v))
                elif "time_to_end" in name: # Relative noise
                    post_obs_piece = obs_piece * (1 + torch_rand_float(-noise_scale, noise_scale, obs_piece.shape, device=self.device))
                else: # Absolute noise
                    post_obs_piece = obs_piece + torch_rand_float(-noise_scale, noise_scale, obs_piece.shape, device=self.device)
            
            obs.append(post_obs_piece)
            self.obs_slice[name] = slice(index, index + len_obs)
            index += len_obs
            
            if self.debug_vis: # for noise range visualization 
                self.states[name] = post_obs_piece
        
        cur_obs = torch.cat(obs, dim=-1)
        
        return cur_obs
    

    def _stacking_states(self, state_names):
        states = []
        index = 0
        for name in state_names:
            state_piece = self.states[name]
            len_state = state_piece.shape[-1]
            states.append(state_piece)
            self.state_slice[name] = slice(index, index + len_state)
            index += len_state
        
        cur_state = torch.cat(states, dim=-1)
        return cur_state
    
    
    def maskout_buf(self, buf, buf_slice):
        """Mask out the buffer that are not used in the training."""
        for name in self.maskout_names:
            if name in buf_slice:
                buf[:, buf_slice[name]] = 1.
        return buf
    

    def maskout_all_timeaware(self, buf, buf_slice):
        """Mask out all time-aware observations."""
        for name in self.maskout_names_full:
            if name in buf_slice:
                buf[:, buf_slice[name]] = 1.
        return buf
    

    def update_memory_buf(self, obs_queue, cur_obs):
        # Remove the oldest observation and add the new one
        obs_queue.pop(0)
        obs_queue.append(cur_obs)
        obs_buf = torch.cat(obs_queue, dim=-1)
        return obs_buf
    

    def get_states_dict(self, real=False):
        """Get the states buffer for the environment."""
        states = self.states if not real else self.states_real
        prev_states = self.prev_states if not real else self.prev_states_real
        return states, prev_states
    

    def update_time_ratio_buf(self, new_time_ratio, env_ids=None):
        env_ids = env_ids if env_ids is not None else self.all_env_ids
        self.prev_time_ratio_buf[env_ids] = self.time_ratio_buf[env_ids].clone()
        self.time_ratio_buf[env_ids] = new_time_ratio
    
    
    def update_linvel_gt(self, env_ids=None):
        env_ids = env_ids if env_ids is not None else self.all_env_ids
        scevelSchedule = self.cfg.get("scevelSchedule", 1.0)
        if self.cfg.get("exp_scheduler", False):
            # Exponential scheduler
            linvel_scaler = self.time_ratio_buf[env_ids] ** scevelSchedule
        else:
            # Linear scheduler
            linvel_scaler = self.time_ratio_buf[env_ids] * scevelSchedule
        self.linvel_max_gt[env_ids] = linvel_scaler * self.linvel_max_gt_init[env_ids]


    def record_init_configs(self, env_ids):
        if "init_configs" not in self.extras:
            self.extras["init_configs"] = {}
            init_configs = self.extras["init_configs"]
            init_configs["time_used"] = []
            init_configs["max_linvel"] = []
            init_configs["sum_linvel"] = []
            self._init_task_configs_buf()
        
        init_configs = self.extras["init_configs"]
        cur_index = len(init_configs["time_used"])
        self._record_task_init_configs(env_ids)
        init_configs["time_used"].extend([0] * len(env_ids)) # Just extend with 0s for index later
        init_configs["max_linvel"].extend([0] * len(env_ids))
        init_configs["sum_linvel"].extend([0] * len(env_ids))
        self.time_used_accm.extend([0] * len(env_ids))
        self.max_linvel_accm.extend([0] * len(env_ids))
        self.sum_linvel_accm.extend([0] * len(env_ids))
        new_env2index = torch.arange(cur_index, cur_index + len(env_ids), dtype=torch.long, device=self.device)
        self.env2index[env_ids] = new_env2index
    
    
    def record_post_config(self, success_ids):
        if "init_configs" not in self.extras: # Maybe warmup stage
            return
        init_configs = self.extras["init_configs"]
        buffer_index = self.env2index[success_ids]
        batch_time_cur = self.time_cur_buf[success_ids].cpu().tolist()
        batch_max_linvel = self.max_linvel_buf[success_ids].cpu().tolist()
        batch_sum_linvel = self.accm_instability[success_ids].cpu().tolist()
        for i in range(len(success_ids)):
            success_id = success_ids[i]
            if buffer_index[i] == -1: continue # This env buffer has not been initialized
            if self.cfg.get("update_configs", False):
                if success_id >= self.num_configs: continue # Out of recorded envs
                if success_id in self.saved_eps: continue # Already saved envs
            self.success_counts[success_id] += 1
            self.time_used_accm[buffer_index[i]] += batch_time_cur[i]
            self.max_linvel_accm[buffer_index[i]] += batch_max_linvel[i]
            self.sum_linvel_accm[buffer_index[i]] += batch_sum_linvel[i]
            if self.success_counts[success_ids[i]] >= self.save_threshold:
                init_configs["time_used"][buffer_index[i]] = self.time_used_accm[buffer_index[i]] / self.success_counts[success_id].item()
                init_configs["max_linvel"][buffer_index[i]] = self.max_linvel_accm[buffer_index[i]] / self.success_counts[success_id].item()
                init_configs["sum_linvel"][buffer_index[i]] = self.sum_linvel_accm[buffer_index[i]] / self.success_counts[success_id].item()
                self.saved_eps.append(success_id)
        self.extras["num_eps_recorded"] = len(self.saved_eps)
        if self.cfg.get("update_configs", False):
            self.extras["update_done"] = len(self.saved_eps) == self.num_configs


    #######################################
    ######### Actions Related #############
    #######################################
    def convert_actions(self, actions):
        # actions have been clipped to be within [-1, 1]
        actions = actions.to(self.device)
            
        if self.cfg.get("meta_rl", False):
            # TODO: add meta-rl support still working on it
            # Convert beta [0, 1] action to [-1, 1]
            actions = actions * 2.0 - 1.0
            cur_obs = self.obs_dict["obs"]
            time_ratio = actions * self.obs_range["time_ratio"].max() # Mapping [-4, 4]
            cur_obs[time_ratio>1, self.time2end_idx] -= self.ctrl_dt * time_ratio[time_ratio>1]
            cur_obs[time_ratio<-1, self.time2end_idx] -= self.ctrl_dt / time_ratio[time_ratio<-1].abs()
            actions, _ = self.pl_agent.get_action_and_value(cur_obs, action_only=True)

        if self.use_beta:
            # Convert beta [0, 1] action to [-1, 1]
            actions = actions * 2.0 - 1.0

        if self.training and self.add_act_noise:
            # Add action noise
            actions[:, :-1] = actions[:, :-1] * \
                              (1 + torch_rand_float(-1, 1., actions[:, :-1].shape, device=self.device) * self.cur_dr_params["noise"]["action"])
            actions[:, :-1] = torch.clamp(actions[:, :-1], -1., 1.)
        
        if self.ratio_range is not None or self.goal_speed is not None:
            if self.time_ratio_buf.min() < 1. and self.cfg.get("scale_actions", False): # slow down motion need to scale
                scale_env_idx = self.time_ratio_buf < 1.
                actions[scale_env_idx, :-1] *= self.time_ratio_buf[scale_env_idx].unsqueeze(-1)

        self.raw_actions = actions.clone()
        self.actions = self.raw_actions.clone()

        return self.raw_actions


    def deploy_joint_command(self, index=-1):
        if self.control_type == "ik":
            self._arm_pos_control[:] = self.tgt_q_seq[index]
            self._gripper_control[:] = self.tgt_gripper_seq[index]
        # Deploy actions
        self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(self._pos_control))
        self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self._effort_control))


    def command_arm(self):
        # Split arm and gripper command
        dpose = self.actions[:, :-1]
        dpose = dpose * self.cmd_limit * self.act_scale

        # Control arm
        if self.control_type == "osc":
            u_arm = self._compute_osc_torques(dpose=dpose)
            self._arm_control[:] = u_arm
        elif self.control_type == "ik":
            u_arm = self._differential_ik(dpose=dpose)
            # self._arm_pos_control[:] = u_arm
        elif self.control_type == "jp":
            u_arm = self._joint_fk(dpose=dpose)
            self._arm_pos_control[:] = u_arm
        else:
            raise ValueError(f"Unknown control type: {self.control_type}")
        
    
    def command_gripper(self):
        u_gripper = self.actions[:, -1]
        # Control gripper
        self.gripper_ctrl_counts += 1
        gripper_ctrl_idx = self.gripper_ctrl_counts >= self.gripper_freq_inv
        if gripper_ctrl_idx.any():
            gripper_ctrl_ids = gripper_ctrl_idx.nonzero().flatten()
            self.gripper_ctrl_counts[gripper_ctrl_idx] = 0
            
            # u_gripper > 0 is close, u_gripper <= 0 is open
            cur_gripper_mode_temp = torch.where(u_gripper[gripper_ctrl_idx]>0., 1., -1.)
            # find the different idx with the current gripper mode
            diff_idx = (cur_gripper_mode_temp != self._gripper_mode_temp[gripper_ctrl_idx])
            diff_ids = gripper_ctrl_ids[diff_idx]
            # If the gripper mode is changed, we need to reset the delay timer. For ids we can not use .any()!!
            if len(diff_ids)>0:
                delay_time = self.cur_dr_params["controller"]["gripper_delay"] + \
                             torch_rand_float(-1, 1, (len(diff_ids), 1), device=self.device) * self.cur_dr_params["noise"]["gripper_delay"]
                delay_steps = (torch.clamp(delay_time, min=0.) / self.ctrl_dt).ceil().flatten()
                self.gripper_delay_timer[diff_ids] = delay_steps
            # Update the gripper mode temp
            self._gripper_mode_temp[gripper_ctrl_idx] = cur_gripper_mode_temp
        
        self.gripper_delay_timer -= 1
        self.gripper_delay_timer = torch.clamp(self.gripper_delay_timer, min=0)
        delay_done_idx = self.gripper_delay_timer == 0
        if delay_done_idx.any():
            # Delay has been done, we can update the gripper mode
            self._gripper_mode[delay_done_idx] = self._gripper_mode_temp[delay_done_idx]
        
        v_gripper = self._real_franka_velocity_limits[-2:].unsqueeze(0)
        if self.add_act_noise:
            # Add gripper velocity noise; both dof share same noise
            v_gripper = v_gripper + torch_rand_float(-1, 1., (self.num_envs, 1), device=self.device) * self.cur_dr_params["noise"]["v_gripper"]
        
        u_fingers = torch.where(self._gripper_mode==1.,
                                -v_gripper[:, 0] * self.ctrl_dt,
                                v_gripper[:, 1] * self.ctrl_dt)
        
        p_fingers = self.prev_tgtq_gripper + u_fingers.unsqueeze(1)
        p_fingers = tensor_clamp(p_fingers, 
                                 self.franka_dof_lower_limits[-2:].unsqueeze(0), 
                                 self.franka_dof_upper_limits[-2:].unsqueeze(0))
        self.tgt_gripper_seq = [self.prev_tgtq_gripper] * (self.num_inter_steps-1) + [p_fingers]  # Update the target gripper sequence with the final position
        # Write gripper command to appropriate tensor buffer
        self.prev_tgtq_gripper[:] = p_fingers
    
    
    def _unify_quat_states(self, real=False):
        """
        Unify the quaternion states to avoid ambiguity.
        We make sure the abs value of largest element of the quaternion is positive to avoid any ambiguity.
        """
        # Unify the quaternion here to remove the ambiguity
        states, _ = self.get_states_dict(real=real)
        unify_quat_func = self.unify_quat if not real else self.unify_quat_np
        for key in states.keys():
            if "quat" in key:
                states[key] = unify_quat_func(states[key])
    
    
    def unify_quat(self, quat):
        """
        We make sure the abs value of largest element of the quaternion is positive to avoid any ambiguity.
        quat: (N, 4); (x, y, z, w)
        """
        max_idx = torch.argmax(torch.abs(quat), dim=-1)
        sign = torch.sign(quat[torch.arange(len(quat)), max_idx])
        quat = quat * sign.unsqueeze(-1)
        return quat
    

    def unify_quat_np(self, quat):
        """
        numpy version of unify_quat
        We make sure the abs value of largest element of the quaternion is positive to avoid any ambiguity.
        quat: (4, ); (x, y, z, w)
        """
        if isinstance(quat, torch.Tensor):
            quat = quat.cpu().numpy()
        elif not isinstance(quat, np.ndarray):
            quat = np.array(quat)

        max_idx = np.argmax(np.abs(quat))
        sign = np.sign(quat[max_idx])
        quat = quat * sign
        return quat


    ##########################################
    ######### Controller Related #############
    ##########################################
    def _compute_osc_torques(self, dpose, eef_vel=None, q=None, qd=None, j_eef=None, mm=None, kp=None, kd=None, kp_null=None, kd_null=None):
        # Solve for Operational Space Control # Paper: khatib.stanford.edu/publications/pdfs/Khatib_1987_RA.pdf
        # Helpful resource: studywolf.wordpress.com/2013/09/17/robot-control-4-operation-space-control/
        mm = self._mm if mm is None else mm
        j_eef = self._j_eef if j_eef is None else j_eef
        q = self._q[:, :7] if q is None else q
        qd = self._qd[:, :7] if qd is None else qd
        eef_vel = self.states["eef_vel"] if eef_vel is None else eef_vel
        kp = self.kp if kp is None else kp
        kd = self.kd if kd is None else kd
        kp_null = self.kp_null if kp_null is None else kp_null
        kd_null = self.kd_null if kd_null is None else kd_null

        if self.add_act_noise:
            # Add inertial noise
            mm = mm * (1 + torch_rand_float_3d(-1, 1., mm.shape, device=self.device) * self.cur_dr_params["noise"].get("inertial_mat", 0.0))
        
        mm_inv = torch.inverse(mm)
        m_eef_inv = j_eef @ mm_inv @ torch.transpose(j_eef, 1, 2)
        m_eef = torch.inverse(m_eef_inv)

        # Transform our cartesian action `dpose` into joint torques `u`
        u = torch.transpose(j_eef, 1, 2) @ m_eef @ (kp * dpose - kd * eef_vel).unsqueeze(-1)

        # Nullspace control torques `u_null` prevents large changes in joint configuration
        # They are added into the nullspace of OSC so that the end effector orientation remains constant
        # roboticsproceedings.org/rss07/p31.pdf
        # https://studywolf.wordpress.com/2013/09/17/robot-control-5-controlling-in-the-null-space/
        j_eef_inv = m_eef @ j_eef @ mm_inv
        u_null = kp_null * ((self.franka_default_dof_pos[:7] - q + np.pi) % (2 * np.pi) - np.pi) - kd_null * qd 
        u_null = mm @ u_null.unsqueeze(-1)
        u_null = (torch.eye(7, device=self.device).unsqueeze(0) - torch.transpose(j_eef, 1, 2) @ j_eef_inv) @ u_null
        u += u_null

        # Clip the values to be within valid effort range
        u = tensor_clamp(u.squeeze(-1),
                         -self._franka_effort_limits[:7].unsqueeze(0), 
                         self._franka_effort_limits[:7].unsqueeze(0))
        
        # if not self.training:
        #     self.print_efforts(u[0])

        return u
    

    def _compute_tsi_torques_with_null(self, dpose):
        # Desired task-space force using PD law (first half is the spring force while the last half is the damping force)
        F_eef = self.kp * dpose - self.kd * self.states["eef_vel"]
        # Clip the values to be within valid effort range
        u = torch.transpose(self._j_eef, 1, 2) @ F_eef.unsqueeze(-1)
        
        q, qd = self._q[:, :7], self._qd[:, :7]
        # pseudo-inverse of the jacobian; torch.linalg.pinv Takes huge amount of time to compute (~0.2s for 1024 envs)
        # j_eef_inv = torch.linalg.pinv(self._j_eef, rcond=1e-2)
        j_eef_inv = pinv_analytical(self._j_eef)
        F_null = self.kp_null * ((self.franka_default_dof_pos[:7] - q + np.pi) % (2 * np.pi) - np.pi) - self.kd_null * qd
        # Compute the null space projector: N = I - J_pinv * J (7x7)
        N_mat = torch.eye(7, device=self.device).unsqueeze(0) - j_eef_inv @ self._j_eef
        u_null = N_mat @ F_null.unsqueeze(-1)
        u += u_null

        u = tensor_clamp(u.squeeze(-1),
                         -self._franka_effort_limits[:7].unsqueeze(0), 
                         self._franka_effort_limits[:7].unsqueeze(0))

        return u


    def _compute_tsi_torques(self, dpose):
        # Desired task-space force using PD law (first half is the spring force while the last half is the damping force)
        F_eef = self.kp * dpose - self.kd * self.states["eef_vel"]
        # Clip the values to be within valid effort range
        u = torch.transpose(self._j_eef, 1, 2) @ F_eef.unsqueeze(-1)
        u = tensor_clamp(u.squeeze(-1),
                         -self._franka_effort_limits[:7].unsqueeze(0), 
                         self._franka_effort_limits[:7].unsqueeze(0))
        # self.print_efforts(u[0])
        return u
    

    def _differential_ik(self, dpose, q=None, j_eef=None, verbose=False):
        """
        Differential IK controller that converts a 6DOF pose error into delta joint angles.
        
        Inputs:
          dpose: Tensor of shape (batch_size, 6) representing the desired 6DOF pose change.
        
        Output:
          dq: Tensor of shape (batch_size, 7) representing the change in joint angles,
              to be used as a delta position command.
        """
        q = self._q[:, :7] if q is None else q
        j_eef = self._j_eef if j_eef is None else j_eef
        # Compute the pseudo-inverse of the end-effector Jacobian.
        # (Using an analytical version for efficiency.)
        j_eef_inv = pinv_analytical(j_eef, eps=0.0025)  # Expected output shape: (batch, 7, 6)
        # Compute the primary delta joint angles using differential IK law: Δq = J† * dpose.
        dq = (j_eef_inv @ dpose.unsqueeze(-1)).squeeze(-1)  # Shape: (batch, 7)
        dq_max_abs = self._real_franka_velocity_limits[:7].unsqueeze(0) * self.ctrl_dt
        dq = tensor_clamp(dq, -dq_max_abs, dq_max_abs)
        
        # Clamp the dq changes to be within safe limits.
        dq = ema_filter(dq, self.prev_dq, alpha=self.cur_dr_params["controller"]["alpha"])
        dq = tensor_clamp(dq, -dq_max_abs, dq_max_abs)
        self.prev_dq[:] = dq.clone()
        
        if not self.training and self.cfg.get("not_move", False):
            dq[:] = 0.
            self.prev_dq[:] = 0.

        d_dq = dq / self.num_inter_steps
        self.tgt_q_seq = []
        for j in range(1, self.num_inter_steps + 1):
            step_tgt_q = self.prev_tgtq + j * d_dq
            step_tgt_q = tensor_clamp(step_tgt_q, q - dq_max_abs, q + dq_max_abs)
            step_tgt_q = tensor_clamp(step_tgt_q,
                                      self.franka_dof_lower_limits[:7].unsqueeze(0),
                                      self.franka_dof_upper_limits[:7].unsqueeze(0))
            self.tgt_q_seq.append(step_tgt_q)
        tgt_q = self.tgt_q_seq[-1]  # Use the last target joint angles as the final target
        self.prev_tgtq[:] = tgt_q.clone()
        
        if verbose:
            # Print the joint angles and the target joint angles
            print(f"Current Joint angles: {q[0]}")
            print(f"Target joint angles: {tgt_q[0]}")
            print(f"Delta joint angles: {dq[0]}")
            print(f"Joint velocity limits: {self._real_franka_velocity_limits[:7]}")
        
        return tgt_q
    

    def _joint_fk(self, dpose, q=None, delta=False):
        """
        Forward Kinematics to compute the end-effector pose given joint angles.
        
        Inputs:
          q: Tensor of shape (batch_size, 7) representing the joint angles.
        
        Output:
          eef_pos: Tensor of shape (batch_size, 3) representing the end-effector position.
          eef_quat: Tensor of shape (batch_size, 4) representing the end-effector orientation as a quaternion.
        """
        # Compute forward kinematics using the robot's URDF model.
        q = self._q[:, :7] if q is None else q
        dq_max_abs = self._real_franka_velocity_limits[:7].unsqueeze(0) * self.ctrl_dt
        dq = dpose * dq_max_abs
        if delta:
            return dq

        dq = ema_filter(dq, self.prev_dq, alpha=self.cur_dr_params["controller"]["alpha"])
        dq = tensor_clamp(dq, dq_max_abs, dq_max_abs)
        self.prev_dq[:] = dq.clone()

        if not self.training and self.cfg.get("not_move", False):
            dq[:] = 0.
            self.prev_dq[:] = 0.

        tgt_q = self.prev_tgtq + dq
        tgt_q = tensor_clamp(tgt_q, q - dq_max_abs, q + dq_max_abs)
        tgt_q = tensor_clamp(tgt_q,
                             self.franka_dof_lower_limits[:7].unsqueeze(0),
                             self.franka_dof_upper_limits[:7].unsqueeze(0))
        self.prev_tgtq[:] = tgt_q.clone()
        
        return tgt_q
    

    def _joint_pd_torque(self, dq, qd):
        u = self.kp_jp * dq - self.kd_jp * qd
        u = tensor_clamp(u,
                         -self._franka_effort_limits[:7].unsqueeze(0), 
                         self._franka_effort_limits[:7].unsqueeze(0))
        return u
    

    #######################################
    ######### RealRobot Related #############
    #######################################
    def _compute_init_config_features(self, data_dict):
        pass


    def _time_related_state_names(self):
        pass

    
    def _reset_timeaware_states_real(self, init_t2e):
        if self.goal_speed is not None:
            self.time_ratio_buf[:] = self.goal_speed
        if self.goal_time is not None:
            self.time_ratio_buf[:] = init_t2e / self.goal_time

        if self.cfg.get("time_ratio", False):
            self.time2end_init[:] = init_t2e
            self.real_time2end_init[:] = init_t2e / self.time_ratio_buf
        else:
            self.time2end_init[:] = init_t2e / self.time_ratio_buf
            self.real_time2end_init[:] = self.time2end_init

        self._reset_bufs(self.all_env_ids)
    
    
    def _estimate_minimum_time2end_real(
            self,
            data_dict,
            new_config,
            k: int = 5,
            use_normalization: bool = False,
        ):
        """
        Estimate minimum time using k-NN with distance ratio-based scaling.
        
        The idea is that if the new configuration has objects closer together than
        a reference configuration, it should take proportionally less time.
        
        Args:
            data_dict: Dictionary with entries containing configuration and min_time
            new_config: New configuration to estimate
            k: Number of nearest neighbors to consider
            use_normalization: Whether to normalize features before distance calculation
            position_weight: Weight for position differences
            orientation_weight: Weight for orientation differences  
            joint_weight: Weight for joint angle differences
            reference_config_key: Key for reference configuration (if None, uses centroid)
            
        Returns:
            Tuple of (estimated_time, info_dict)
        """
        # Extract features from all configurations
        features_array = self._compute_init_config_features(data_dict).cpu().numpy()
        times_array = self.env_configs["time_used"].cpu().numpy()
        
        # Extract features from new configuration
        new_features = self.extract_features(new_config, self._time_related_state_names())
        
        if use_normalization:
            # Normalize features
            scaler_features = StandardScaler()
            features_array = scaler_features.fit_transform(features_array)
            new_features = scaler_features.transform(new_features.reshape(1, -1)).flatten()
        
        # Calculate distances to all points
        distances = self.calculate_weighted_distances(new_features, features_array)
        
        # Find k nearest neighbors; All elements before position k are less than or equal to that k-th smallest element.
        k_nearest_indices = np.argpartition(distances, k)[:k]
        k_nearest_distances = distances[k_nearest_indices]
        k_nearest_times = times_array[k_nearest_indices]
        k_nearest_features = features_array[k_nearest_indices]
        
        # Sort by distance
        sort_indices = np.argsort(k_nearest_distances)
        k_nearest_indices = k_nearest_indices[sort_indices]
        k_nearest_distances = k_nearest_distances[sort_indices]
        k_nearest_times = k_nearest_times[sort_indices]
        k_nearest_features = k_nearest_features[sort_indices]
        
        # Calculate weighted average of ratio-based estimates
        estimated_time = np.mean(k_nearest_times).item()

        print(f"\nEstimate Real World Time2End: {estimated_time:.3f}s using k-NN with k={k}\n")
        
        return estimated_time
    
    
    def _validate_robot_state_consistency(self, franka_states):
        """Validate consistency between simulation and real robot states."""
        sim_franka_dof = self.fk_init_dof[0, :7].cpu().numpy()
        real_franka_dof = franka_states["q"]
        diff = np.abs(sim_franka_dof - real_franka_dof)
        
        if np.max(diff) > 0.01:
            raise ValueError(
                f"Simulation/Real robot state mismatch. Max diff: {np.max(diff):.4f}\n"
                f"Sim DOF: {sim_franka_dof}\n"
                f"Real DOF: {real_franka_dof}"
            )
    
    
    def _init_real_robot_mode(self, state_estimator, franka_arm):
        """Initialize real robot mode."""
        self._update_states_real(state_estimator, franka_arm, max_trials=3)
        if hasattr(self, "env_configs") and self.env_configs is not None:
            estimated_time = self._estimate_minimum_time2end_real(self.env_configs, self.states_real)
            self._reset_timeaware_states_real(estimated_time)
        obs_buf, extras = self.compute_observations_real(state_estimator, franka_arm)
        
        # Validate initial joint positions
        self._validate_robot_state_consistency(self.states_real)
        
        return obs_buf, extras
    
    
    def _update_task_states_real(self):
        pass


    def _compensate_task_states_real(self, obj_states, arm_states):
        return obj_states
    
    
    def _get_robot_state_with_retry(self, franka_arm, max_attempts=100):
        """Get robot state with retry logic."""
        for _ in range(max_attempts):
            franka_states = franka_arm.get_state()
            if franka_states is not None:
                return franka_states
        
        # Allow None state for pure simulation without FK replay
        if self.cfg.get("use_sim_pure", False) and not self.cfg.get("use_fk_replay", False):
            return None
        
        raise ValueError(f"Failed to get robot state after {max_attempts} attempts")


    def _update_robot_states_real(self, franka_arm):
        # Get the franka states (from the real robot or the simulation)
        franka_states = {}
        franka_states_real = self._get_robot_state_with_retry(franka_arm, max_attempts=100)
        if franka_states_real is None:
            return None
        
        franka_states.update({
            "eef_pos": np.array(franka_states_real["eef_pos"]),
            "eef_quat": np.array(franka_states_real["eef_quat"]),
            "q": franka_states_real["q"],
            "qd": franka_states_real["qd"],
            "q_gripper": franka_states_real["q_gripper"],
            "eef_vel": franka_states_real["eef_vel"],
            "mm": np.array(franka_states_real["mm"]).reshape(7, 7),
            "j_eef": np.array(franka_states_real["j_eef"]).reshape(6, 7),

            "prev_tgtq": self.prev_tgtq[0].cpu().numpy(),
            "prev_dq": self.prev_dq[0].cpu().numpy(),
            "gripper_mode": self._gripper_mode_temp[0].cpu().numpy(),
        })

        return franka_states
    

    def _get_states_with_retry(self, state_estimator, franka_arm, max_trials):
        """Get object and arm states with retry logic."""
        for i in range(max_trials):
            obj_states = self._update_task_states_real(state_estimator)
            arm_states = self._update_robot_states_real(franka_arm)
            obj_states = self._compensate_task_states_real(obj_states, arm_states)
            if obj_states is not None and arm_states is not None:
                return obj_states, arm_states
        
        # Raise specific errors for debugging
        if obj_states is None:
            raise TimeoutError(
                f"Failed to get cube poses after {max_trials} attempts. "
                "Please check if cubes are in camera view."
            )
        if arm_states is None:
            raise TimeoutError(
                f"Failed to get robot state after {max_trials} attempts. "
                "Please check robot connection."
            )
        

    def _update_states_real(self, state_estimator, franka_arm, max_trials, reset_ids=[]):
        # Update the states that require the computation from both current states and previous states
        # TODO: make all states to torch at the beginning since we finally need all states to be torch (at least one transfer)
        obj_states, arm_states = self._get_states_with_retry(
            state_estimator, franka_arm, max_trials
        )
        self.states_real.update({
            **obj_states,
            **arm_states,
        })

        # manually reset the previous states here; because issacgym need one more step to refresh the states
        # self._reset_prev_states_real(reset_ids)
        self._update_diff_states(real=True)
        self._update_timeaware_states(real=True)
        self._update_prev_states(real=True)
        # Unify the quaternion here to remove the ambuity
        self._unify_quat_states(real=True)

        self._update_common_info_real()
        self._update_debug_info_real()
    

    def _stacking_obs_real(self, obs_names):
        obs = []
        index = 0
        for name in obs_names:
            obs_piece = to_numpy(self.states_real[name])
            if len(obs_piece.shape) == 0:  # If it is a numpy scalar
                obs_piece = np.expand_dims(obs_piece, axis=0)
            len_obs = obs_piece.shape[-1]
            obs.append(obs_piece)
            self.obs_slice[name] = slice(index, index + len_obs)
            index += len_obs
        return to_torch(np.concatenate(obs), device=self.device).unsqueeze(0)
    
    
    def _collect_process_obs_real(self):
        """Collect observations and update buffers."""
        # Get observation names
        obs_names = self.get_taskobs_names()
        obs_names = self.add_timeaware_obs(obs_names)
        
        # Collect current observations
        cur_obs = self._stacking_obs_real(obs_names)
        cur_obs = self.maskout_buf(cur_obs, self.obs_slice)
        
        # Update memory buffer
        obs_buf = self.update_memory_buf(self.real_obs_queue, cur_obs)
        self.prev_obs[:] = cur_obs
        
        return obs_buf
    
    
    def compute_observations_real(self, state_estimator, franka_arm, max_trials=3):
        """Compute observations from real world sensors."""
        # _update_states
        self._update_states_real(state_estimator, franka_arm, max_trials=max_trials)
        # Collect and process observations
        obs_buf = self._collect_process_obs_real()
        
        # Visualization
        self.map_real2sim()
        
        return obs_buf, self.extras


    def _check_time_termination(self):
        """Check if episode should terminate based on time."""
        if self.goal_speed is None and self.goal_time is None:
            return False
        
        obs_time2end = self.time_to_end[0]
        grace_time = 5 if self.cfg.get("task_name", "") == "FrankaGmPour" else 2
        done = (obs_time2end <= -grace_time)
        
        return done
    
    
    def compute_reward_real(self):
        done = self._check_time_termination()
        reward = 0.0
        success = False
        
        # Convert to torch tensors
        return (
            to_torch([reward], device=self.device),
            to_torch([done], device=self.device),
            to_torch([success], device=self.device)
        )
    

    def pre_physics_step_real(self, actions):
        self.convert_actions(actions)
        u_arm = self.command_arm_real().flatten()
        u_gripper = self.command_gripper_real().flatten()
        u = torch.cat([u_arm, u_gripper], dim=-1)
        
        return u.cpu().tolist()
    

    def command_arm_real(self):
        # Split arm and gripper command (keep the dim)
        dpose = self.actions[:, :-1]
        # Scale arm value first
        dpose = dpose * self.cmd_limit * self.act_scale

        states = self.states_real
        if self.control_type == "osc":
            eef_vel = to_torch(states["eef_vel"], device=self.device).unsqueeze(0)
            q = to_torch(states["q"], device=self.device).unsqueeze(0)
            qd = to_torch(states["qd"], device=self.device).unsqueeze(0)
            mm = to_torch(states["mm"], device=self.device).unsqueeze(0)
            j_eef = to_torch(states["j_eef"], device=self.device).unsqueeze(0)
            u_arm = self._compute_osc_torques(
                dpose=dpose, 
                eef_vel=eef_vel, 
                q=q, 
                qd=qd, 
                j_eef=j_eef, 
                mm=mm,
                kp=self.kp_real,
                kd=self.kd_real,
                kp_null=self.kp_null_real,
                kd_null=self.kd_null_real,
            )
            self._arm_control[:] = u_arm

        elif self.control_type == "ik":
            q = to_torch(states["q"], device=self.device).unsqueeze(0)
            j_eef = to_torch(states["j_eef"], device=self.device).unsqueeze(0)
            u_arm = self._differential_ik(dpose=dpose, q=q, j_eef=j_eef)

        elif self.control_type == "jp":
            q = to_torch(states["q"], device=self.device).unsqueeze(0)
            u_arm = self._joint_fk(dpose=dpose, q=q)
            self._arm_pos_control[:] = u_arm
        
        return u_arm


    def command_gripper_real(self):
        raw_gripper = self.actions[:, -1]
        # Gripper control
        self.gripper_ctrl_counts_real += 1
        if self.gripper_ctrl_counts_real >= self.gripper_freq_inv:
            self.gripper_ctrl_counts_real = 0
            cmd_gripper = torch.where(raw_gripper > 0., 1., -1.)
            u_gripper = torch.where(cmd_gripper==self.prev_u_gripper_real, 0., cmd_gripper)
            self.prev_u_gripper_real = self._gripper_mode[0] = self._gripper_mode_temp[0] = cmd_gripper
        else:
            u_gripper = torch.zeros_like(raw_gripper)

        return u_gripper
    
    
    #----------------- Info Logging -----------------#
    def _update_debug_info_real(self):
        pass
    
    
    def _update_common_info_real(self):
        # Update the common info for the real robot
        self.extras.update({
            # Time related
            "observed_time2end": self.time_to_end[0].clone(),
            "real_time2end": self.real_time_to_end[0].clone(),
            "eps_time_goal": self.real_time2end_init.clone(),
            "time_ratio": self.time_ratio_buf.clone(),
            # Scene related
            "scene_linvel": self.states_real["sce_linvel"],
            "scene_linvel_lim": self.states_real["lim_linvel"], # this is used for visualization
            # Franka Related
            "joint_tgt_q": self._arm_pos_control[0].clone(),
            "joint_torqs": self._effort_control[0].clone(),
            "joint_poss": self.states_real["q"],
            "joint_vels": self.states_real["qd"], # Sensor reading velocity is kind of noisy!
            "joint_gripper_poss": self.states_real["q_gripper"],
            "joint_velocity_limits": self._real_franka_velocity_limits.clone(),
        })


    def update_memory_buf_real(self, cur_obs):
        # Remove the oldest observation and add the new one
        self.real_obs_queue.pop(0)
        self.real_obs_queue.append(cur_obs)
        obs_buf = torch.cat(self.real_obs_queue, dim=-1)
        return obs_buf
    

    def map_real2sim(self):
        pass


    #######################################
    ######### Utils #############
    #######################################
    def _reset_init_cube_state(self, 
                               cube_name,
                               cube_sizes,
                               surface2cube_z,
                               other_cube_state,
                               other_cube_sizes,
                               env_ids, 
                               check_valid=True):
        """
        Simple method to sample @cube's position based on self.startPositionNoise and self.startRotationNoise, and
        automaticlly reset the pose internally. Populates the appropriate self._init_cubeX_state

        If @check_valid is True, then this will also make sure that the sampled position is not in contact with the
        other cube.

        Args:
            cube(str): Which cube to sample location for. Either 'A' or 'B'
            env_ids (tensor or None): Specific environments to reset cube for
            check_valid (bool): Whether to make sure sampled position is collision-free with the other cube.
        """
        # If env_ids is None, we reset all the envs
        if env_ids is None:
            env_ids = torch.arange(start=0, end=self.num_envs, device=self.device, dtype=torch.long)
        num_resets = len(env_ids)

        # Initialize buffer to hold sampled values
        sampled_cube_state = torch.zeros(num_resets, 13, device=self.device)
        this_l, _, _ = cube_sizes[:, 0], cube_sizes[:, 1], cube_sizes[:, 2]
        other_l, _, _ = other_cube_sizes[:, 0], other_cube_sizes[:, 1], other_cube_sizes[:, 2]

        # Minimum cube distance for guarenteed collision-free sampling is the sum of each cube's effective radius; CubeA size is full length
        min_dists = (this_l + other_l)[env_ids] * np.sqrt(2) * 0.8

        # Sampling is "centered" around middle of table
        sign = 1.0 if "B" in cube_name else -1.0
        initial_xy_offset = torch.zeros((num_resets, 2), device=self.device, dtype=torch.float32)
        initial_xy_offset[:, 1] = sign * torch.max(this_l[env_ids], other_l[env_ids]) * np.sqrt(2)
        centered_cube_xy_state = self._ws_surface_pos[:2] + initial_xy_offset

        # Set z value, which is fixed height
        sampled_cube_state[:, 2] = self._ws_surface_pos[2] + surface2cube_z[env_ids]
        sampled_cube_state[:, 6] = 1.0

        # If we're verifying valid sampling, we need to check and re-sample if any are not collision-free
        # We use a simple heuristic of checking based on cubes' radius to determine if a collision would occur
        success = not check_valid
        # Indexes corresponding to envs we're still actively sampling for
        active_idx = torch.arange(num_resets, device=self.device)
        for i in range(200):
            # Sample x y values
            if self.cfg.get("max_dist", False):
                sampled_delta_xy = sign * self.cur_dr_params["spatial"][f"{cube_name}_pos"] * torch.ones_like(sampled_cube_state[active_idx, :2], device=self.device)
            else:
                sampled_delta_xy = self.cur_dr_params["spatial"][f"{cube_name}_pos"] * 2.0 * (torch.rand_like(sampled_cube_state[active_idx, :2]) - 0.5)
            sampled_cube_state[active_idx, :2] = centered_cube_xy_state[active_idx] + sampled_delta_xy
            sampled_cube_state[active_idx, :2] = torch.clamp(sampled_cube_state[active_idx, :2],
                                                             self._ws_surface_pos[:2]-self._ws_upper_bounds[:2],
                                                             self._ws_surface_pos[:2]+self._ws_upper_bounds[:2])
            # Check if sampled values are valid
            if not check_valid:
                break
            else:
                cube_dist = torch.linalg.norm(sampled_cube_state[:, :2] - other_cube_state[env_ids, :2], dim=-1)
                active_idx = torch.nonzero(cube_dist < min_dists, as_tuple=True)[0]
                # If active idx is empty, then all sampling is valid :D
                if len(active_idx) == 0:
                    success = True
                    break
        # Make sure we succeeded at sampling
        assert success, "Sampling cube locations was unsuccessful! ):"

        # Sample rotation value (Only rotating around z axis!! Watch out the evaluation!)
        if self.cur_dr_params["spatial"][f"{cube_name}_quat"] > 0:
            aa_rot = torch.zeros(num_resets, 3, device=self.device)
            aa_rot[:, 2] = self.cur_dr_params["spatial"][f"{cube_name}_quat"] * 2.0 * (torch.rand(num_resets, device=self.device) - 0.5)
            sampled_cube_state[:, 3:7] = quat_mul(axisangle2quat(aa_rot), sampled_cube_state[:, 3:7])

        # Lastly, set these sampled values as the new init state
        return sampled_cube_state
    
    
    def _reset_timeaware_states(self, new_cfg_env_ids, config_index):
        # Reset speed ratio first
        if self.cfg.get("warmup_rand", False):
            self.time_ratio_buf[new_cfg_env_ids] = torch_rand_float(*self.obs_range["time_ratio"], (1, len(new_cfg_env_ids)), device=self.device)
        elif self.goal_speed is not None:
            self.time_ratio_buf[new_cfg_env_ids] = self.goal_speed
        elif self.ratio_range is not None:
            probs = torch.rand(len(new_cfg_env_ids), device=self.device)
            slowset_env_ids = new_cfg_env_ids[probs <= 0.]
            fastest_env_ids = new_cfg_env_ids[probs >= 1.]
            between_env_ids = new_cfg_env_ids[(probs > 0.) & (probs < 1.)]
            self.time_ratio_buf[slowset_env_ids] = self.ratio_range[0]
            self.time_ratio_buf[fastest_env_ids] = self.ratio_range[1]
            if len(between_env_ids) > 0:
                self.time_ratio_buf[between_env_ids] = torch_rand_float(self.ratio_range[0], self.ratio_range[1], (1, len(between_env_ids)), device=self.device)

        # Reset the time2end and real_time2end
        if self.cfg.get("warmup_rand", False):
            # Pure random during teacher-student warmup
            init_t2e = torch_rand_float(*self.obs_range["time_to_end"], (1, len(new_cfg_env_ids)), device=self.device)
            init_limvel = torch_rand_float(*self.obs_range["max_linvel"], (1, len(new_cfg_env_ids)), device=self.device)
            self._init_t2ebuf_linvelbuf(init_t2e, init_limvel, new_cfg_env_ids)
        elif config_index is not None:
            # Use the configs to reset the time2end and limvel (timeaware training)
            init_t2e = self.env_configs["time_used"][config_index]
            init_limvel = self.env_configs["max_linvel"][config_index]
            if self.goal_time is not None:
                self.time_ratio_buf[new_cfg_env_ids] = init_t2e / self.goal_time
            self._init_t2ebuf_linvelbuf(init_t2e, init_limvel, new_cfg_env_ids)
        else:
            # Pure random during pre training
            self.time2end_init[new_cfg_env_ids] = self.obs_range["time_to_end"][1] # Reset to the max time
            self.real_time2end_init[new_cfg_env_ids] = self.time2end_init[new_cfg_env_ids]
            
    
    def _init_t2ebuf_linvelbuf(self, init_t2e, init_limvel, env_ids):
        if self.cfg.get("time_ratio", False):
            self.time2end_init[env_ids] = init_t2e
            self.real_time2end_init[env_ids] = init_t2e / self.time_ratio_buf[env_ids]
        else:
            self.time2end_init[env_ids] = init_t2e / self.time_ratio_buf[env_ids]
            self.real_time2end_init[env_ids] = self.time2end_init[env_ids]
        
        self.linvel_max_gt_init[env_ids] = init_limvel
        self.update_linvel_gt(env_ids)

    
    def recompute_staged_time_ratio(self, reset_ids=[]):
        if self.use_staged_ctrl and (len(reset_ids) > 0):
            avg_time_ratio = self.time_ratio_buf[reset_ids].clone()
            invalid_avg_time_ratio = (avg_time_ratio < self.ratio_range[0]).any() or (avg_time_ratio > self.ratio_range[1]).any()
            if invalid_avg_time_ratio.any():
                raise ValueError(f"The fastest time usage range is [{self.time2end_init[reset_ids].min()}, {self.time2end_init[reset_ids].max()}], please adjust the goal time to make sure it is large enough")
    
            if self.cfg.get("use_avg_speed", False):
                fast_time_ratio = slow_time_ratio = avg_time_ratio
            else:
                ratio_d_fast_d_slow = self.slow_portion / self.fast_portion
                d_slow_time_ratio = torch.min(avg_time_ratio - self.ratio_range[0], (self.ratio_range[1] - avg_time_ratio) / ratio_d_fast_d_slow)
                d_fast_time_ratio = d_slow_time_ratio * ratio_d_fast_d_slow
                fast_time_ratio = avg_time_ratio + d_fast_time_ratio
                slow_time_ratio = avg_time_ratio - d_slow_time_ratio
            
            for i in range(len(self.budget_portion)):
                if self.speed_describe[i] == 1:
                    self.stage_time_ratio_buf[reset_ids, i] = fast_time_ratio
                else:
                    self.stage_time_ratio_buf[reset_ids, i] = slow_time_ratio
            self.cur_stage[reset_ids] = 0
            

    def _apply_cube_state_noise(self, template_cube_state, env_ids, noise_scale=0.01):
        """
        Apply small spatial random noise to the cube state, when using fixed configurations.
        """
        num_resets = len(env_ids)
        sampled_cube_state = template_cube_state.repeat(num_resets, 1)
        # We just directly sample
        sampled_cube_state[:, :2] = sampled_cube_state[:, :2] + \
                                    2.0 * noise_scale * (
                                    torch.rand(num_resets, 2, device=self.device) - 0.5)
        aa_rot = torch.zeros(num_resets, 3, device=self.device)
        aa_rot[:, 2] = 2.0 * noise_scale * (torch.rand(num_resets, device=self.device) - 0.5)
        sampled_cube_state[:, 3:7] = quat_mul(axisangle2quat(aa_rot), sampled_cube_state[:, 3:7])
        return sampled_cube_state
    

    def apply_disturbances(self):
        states = self.states
        approaching_dist = states["approaching_dist"]
        ready_to_apply = torch.where(approaching_dist < 0.05, 1, 0)
        ready_to_apply = ready_to_apply * (self.force_has_applied == 0)  # Only apply it once
        env_ids = torch.nonzero(ready_to_apply, as_tuple=True)[0]
        forces_directions = torch_rand_float(-1, 1, (len(env_ids), 3), device=self.device)
        forces_directions[:, 2] = 0.
        forces = forces_directions / torch.linalg.norm(forces_directions, dim=-1, keepdim=True) * self.disturbance_v
        self.apply_rigid_body_force(env_ids, forces)
        self.force_has_applied[env_ids] = 1
    
    
    def apply_rigid_body_force(self, env_id, forces, body_handle=None):
        assert len(env_id) == len(forces), "env_id and forces must have the same length"
        body_handle = self.link_handles[self.apply_force_handle] if body_handle is None else body_handle
        apply_forces = torch.zeros((self.num_envs, self.num_bodies, 3), device=self.device, dtype=torch.float)
        apply_forces[env_id, body_handle] = forces
        # Convert forces to the correct shape
        self.gym.apply_rigid_body_force_tensors(self.sim, gymtorch.unwrap_tensor(apply_forces), None, gymapi.ENV_SPACE)

    
    def _update_props(self, env_ids):
        print(f"Warning: _update_props is not implemented. Please implement it if needed.")
        pass
    

    def extract_features(self, config, state_names) -> np.ndarray:
        """
        Extract and concatenate all features from a configuration.
        
        Args:
            config: Dictionary containing:
                - 'cubeA_pos': 3D position (x, y, z) in meters
                - 'cubeA_quat': 4D quaternion (w, x, y, z)
                - 'cubeB_pos': 3D position (x, y, z) in meters
                - 'cubeB_quat': 4D quaternion (w, x, y, z)
                - 'initial_joints': 7D joint angles in radians
                
        Returns:
            Feature vector of length 21 (3+4+3+4+7)
        """
        features = []
        
        for key in state_names:
            features.append(to_numpy(config[key]))

        features = np.concatenate(features, axis=-1).flatten()
        
        return features    
    
    
    def calculate_weighted_distances(
        self,
        new_features: np.ndarray,
        features_array: np.ndarray,
    ) -> np.ndarray:
        """
        Calculate weighted distances between one configuration and multiple configurations.
        
        Args:
            new_features: Single feature vector (shape: (21,))
            features_array: Multiple feature vectors (shape: (N, 21))
            position_weight: Weight for position differences
            orientation_weight: Weight for orientation differences
            joint_weight: Weight for joint angle differences
            
        Returns:
            Array of weighted distances (shape: (N,))
        """
        # Broadcast new_features to match features_array shape
        new_features_broadcast = new_features[np.newaxis, :]
        distances = np.linalg.norm(new_features_broadcast - features_array, axis=1)  # Shape: (N,)
        
        return distances
    
    
    def visualize_configs_seq(self):
        specific_idx = input("Enter the specific index to visualize the configs sequence: ")
        if specific_idx.isdigit():
            specific_idx = int(specific_idx)
        elif specific_idx == "":
            specific_idx = self.cfg["specific_idx"] + 1
        self.cfg["specific_idx"] = specific_idx
        self.reset_all()
        print(f"Current specific index: {self.cfg['specific_idx']}")
    
    
    def get_viewer_image(self, env_id=0):
        color_image = self.gym.get_camera_image(self.sim, self.envs[env_id], self.camera, gymapi.IMAGE_COLOR)
        # Reshape the image to (height, width, 4) and convert to numpy array
        color_image = color_image.reshape(self.camera_h, self.camera_w, 4)
        return color_image

    
    def is_warmup_done(self):
        """
        Check if the warmup is done.
        Return: True if the warmup is done, False otherwise.
        """
        return self.num_episodes >= self.cfg.get("warmup_episodes", 0)
    

    def is_ready_to_record(self, env_ids):
        return self.cfg.get("record_init_configs", False) and \
               self.is_warmup_done() and \
               len(env_ids) > 0

    
    def point2frankaBase(self, quat, pos):
        """
        Convert the point from the world frame to the franka base frame.
        pos: (n_envs, 3), quat: (n_envs, 4).
        Return: quat, pos
        """
        return tf_combine(self._franka2W_quat, self._franka2W_pos, quat, pos)
    
    def vec2frankaBase(self, vec):
        """
        Convert the vector from the world frame to the franka base frame.
        vec: (n_envs, 3).
        Return: vec
        """
        return quat_apply(self._franka2W_quat, vec)
    
    def points2frankaBase(self, pos, quat=None):
        """
        Convert the points from the world frame to the franka base frame.
        pos: (n_envs, n_points, 3).
        Return: pos
        """
        _, m, _ = pos.shape
        _franka2W_quat = self._franka2W_quat.unsqueeze(1).repeat(1, m, 1)
        _franka2W_pos = self._franka2W_pos.unsqueeze(1).repeat(1, m, 1)
        return tf_apply(_franka2W_quat, _franka2W_pos, pos)
    
    def draw_point(self, pos, ori=None):
        # Draw point in the env 0
        sphere_ori, sphere_pos = gymapi.Transform(), gymapi.Transform()
        sphere_ori.r = gymapi.Quat(*ori) if ori is not None else gymapi.Quat(0, 0, 0, 1)
        sphere_pos.p = gymapi.Vec3(*pos)
        sphere_geom = gymutil.WireframeSphereGeometry(0.01, 12, 12, sphere_ori, color=(1, 1, 0))
        gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.envs[0], sphere_pos)

    
    def draw_axes(self, pos, ori=None):
        # Draw axes in the env 0
        axes_ori, axes_pos = gymapi.Transform(), gymapi.Transform()
        axes_ori.r = gymapi.Quat(*ori) if ori is not None else gymapi.Quat(0, 0, 0, 1)
        axes_pos.p = gymapi.Vec3(*pos)
        axes_geom = gymutil.AxesGeometry(0.1, axes_ori)
        gymutil.draw_lines(axes_geom, self.gym, self.viewer, self.envs[0], axes_pos)


    def clear_lines(self):
        self.gym.clear_lines(self.viewer)

    
    def update_debug_info(self, name, value):
        if name not in self.debug_info:
            self.debug_info[name] = []
        self.debug_info[name].append(value)


    @staticmethod
    def same_address(x, y):
        return x.storage().data_ptr() == y.storage().data_ptr()
    

    # Print functions
    def print_efforts(self, env_u):
        print("Efforts: ", end="")
        for i in range(len(env_u)):
            print(f" {i}: {env_u[i]:.3f} | ", end="")
        print()


###############################################################
######### External Class Utils for specpfic tasks #############
###############################################################
def is_under_valid_vel(linvel_norm, vel_limit=10.):
    """
    Check if the object is under the velocity limit
    Input:
        linvel_norm: (num_envs, 1)
        vel_limit: float
    Output:
        in_vel: (num_envs, 1)
    """
    # Check if the object is under the velocity limit
    return linvel_norm <= vel_limit


def is_under_valid_contact(contact_forces_norm, contact_limit=10.):
    """
    Check if the contact forces are under the limit
    Input:
        contact_forces: (num_envs, num_contacts, 3)
        contact_limit: float
    Output:
        in_contact: (num_envs, num_contacts)
    """
    # Check if the contact forces are under the limit
    return contact_forces_norm <= contact_limit


def is_in_cup(gms_pos, cup_pos, cup_rimpos, cup_size):
    """
    Check if the gms is in the cup
    Input:
        gms_pos: (num_envs, num_gms, 3)
        gms_radius: (num_envs, num_gms)
        cup_pos: (num_envs, 3)
        cup_size: (num_envs, 3)
    Output:
        in_cup: (num_envs, num_gms)
    """
    # Check if the gms is in the cup; We need to consider the cup's orientation!
    # Project the gms_pos to the connection line between the cup's bottom and top
    cup_radius, cup_height, cup_thickness = cup_size[:, 0]/2, cup_size[:, 1], cup_size[:, 2]
    if gms_pos.dim() == 2: # gripper pos is in (num_envs, 3)
        gms_pos = gms_pos.unsqueeze(1)
    _, dist, p_scale = project_point_on_segment(cup_pos, cup_rimpos, gms_pos)
    # dist is in (num_envs, num_gms); cup_radius - cup_thickness should also be in (num_envs, 1) to avoid weird broadcasting
    in_radius = dist <= (cup_radius - cup_thickness).unsqueeze(-1)
    in_height = (0 <= p_scale) & (p_scale <= 1)
    
    return in_radius & in_height
    

def project_point_on_segment(x, y, z):
    """
    Projects point z onto the line segment connecting x and y using batch operations.

    Args:
        x: Tensor of shape (num_envs, d), start point of the segment.
        y: Tensor of shape (num_envs, d), end point of the segment.
        z: Tensor of shape (num_envs, n, d), point to project.
    
    Returns:
        proj: Tensor of shape (batch_size, n), the projection of z onto the segment.
    """
    if z.dim() == 2: # z is in (num_envs, d)
        z = z.unsqueeze(1)
    num_gms = z.size(1)
    x, y = x.unsqueeze(1), y.unsqueeze(1)
    # Vector along the segment
    v = (y - x).repeat(1, num_gms, 1) # (num_envs, 1, d) -> (num_envs, n, d)
    # Vector from x to z
    w = z - x # (num_envs, n, d)
    # Squared norm of v
    v_norm_sq = torch.sum(v * v, dim=-1, keepdim=True)  # (num_envs, n, 1)
    # Handle edge case where x == y (degenerate segment)
    v_norm_sq = torch.clamp(v_norm_sq, min=1e-8)  # Avoid division by zero
    # Compute t (projection scalar)
    p_scale = torch.sum(w * v, dim=-1, keepdim=True) / v_norm_sq  # (num_envs, n, 1)
    
    # Compute the projection point
    proj = x + p_scale * v  # (num_envs, n, d)
    dist = torch.norm(z - proj, dim=-1)
    
    return proj, dist, p_scale.squeeze(-1)


def pinv_analytical(j_eef, eps=1e-4):
    # Our assumption is j_eef is a wide and full-row-rank matrix (N, 6, 7)
    # Compute J * J^T for all batches
    jjt = j_eef @ j_eef.transpose(-1, -2)  # shape: (1024, 6, 6)

    # For numerical stability, you can add a small regularization term (optional)
    eye = torch.eye(jjt.shape[-1], device=j_eef.device).expand_as(jjt)
    jjt_reg = jjt + eps * eye

    # Compute the inverse of J*J^T
    jjt_inv = torch.linalg.inv(jjt_reg)  # shape: (1024, 6, 6)

    # Compute the pseudoinverse: J^T * (J * J^T)^(-1)
    j_eef_inv = j_eef.transpose(-1, -2) @ jjt_inv  # shape: (1024, 7, 6)

    return j_eef_inv


def ema_filter(x, prev_x, alpha=0.9):
    """
    Exponential moving average filter.
    x: (num_envs, 1)
    prev_x: (num_envs, 1)
    alpha: float
    """
    return alpha * prev_x + (1 - alpha) * x


def mix_norm(x, dim=-1, keepdim=False):
    if isinstance(x, torch.Tensor):
        return torch.norm(x, dim=dim, keepdim=keepdim)
    elif isinstance(x, np.ndarray):
        return np.linalg.norm(x, axis=dim, keepdims=keepdim)
    

def mix_clone(x):
    if isinstance(x, torch.Tensor):
        return x.clone()
    elif isinstance(x, np.ndarray) or isinstance(x, list):
        return x.copy()
    else:
        raise TypeError(f"Unsupported type {type(x)} for cloning.")
    

def to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.cpu().numpy()
    elif isinstance(x, np.ndarray):
        return x
    else:
        return np.array(x)