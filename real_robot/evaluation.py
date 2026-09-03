"""Real-robot execution lifecycle shared by the main policy evaluator."""

import os
import time
import traceback

import numpy as np
import torch

from core.common.time import remaining_sleep


class RealRobotEvaluationMixin:
    def evaluate_real_robot(self):
        """Run real robot evaluation."""
        from real_robot.StateEstimator import CubePoseEstimator, CupPoseEstimator, DrawerHandlePoseEstimator
        from real_robot.SocketClient import FrankaClient
        from real_robot.DemoCamera import CameraRecorder

        # Initialize hardware
        franka_arm = FrankaClient(
            controller_ip=self.args.controller_ip,
            sub_port=self.args.controller_sub_port,
            pub_port=self.args.controller_pub_port,
        )
        demo_recorder = CameraRecorder(camera_index=self.args.demo_camera_index)

        assert self.args.cam_ext_path is not None

        if self.args.task_name == "FrankaCubeStack":
            state_estimator = CubePoseEstimator(cam_ext_path=self.args.cam_ext_path)
        elif self.args.task_name == "FrankaGmPour":
            state_estimator = CupPoseEstimator(cam_ext_path=self.args.cam_ext_path)
        elif self.args.task_name == "FrankaCabinet":
            state_estimator = DrawerHandlePoseEstimator(cam_ext_path=self.args.cam_ext_path)
        else:
            raise Exception("Not implemented yet")

        # Validate controller
        assert self.envs.cur_dr_params["controller"]["max_vel_subtract"] >= 0.7

        # Configure episode
        episode_length = 60 if self.args.not_move else 1000
        msg_send_count = 0

        # Load action replay buffer if needed
        act_replay_buf = None
        if self.args.debug_act:
            act_file_path = os.path.join("cal_results", "debug_act/sim_act_record.npy")
            if os.path.exists(act_file_path):
                act_replay_buf = torch.tensor(np.load(act_file_path), dtype=torch.float32).to(self.device)
            if not self.args.use_sim_pure:
                assert act_replay_buf is not None and len(act_replay_buf) > 0
                episode_length = len(act_replay_buf)

        # Start recording
        recording_demo = not self.args.use_sim_pure and self.args.demo_name is not None
        if recording_demo:
            os.makedirs(self.args.demo_dir, exist_ok=True)
            output_filename = os.path.join(self.args.demo_dir, self.args.demo_name)
            demo_recorder.start(show_video=False, record_video=True, output_filename=output_filename)

        # Initialize data collection
        sim_robot_dict = {"obs": [], "action": [], "joint_q": []}
        real_robot_dict = {"obs": [], "action": [], "joint_q": []}

        # Main execution loop
        with torch.no_grad():
            self.agent.deterministic = True

            try:
                eps_start_time = time.perf_counter()
                next_obs, infos_real = self.envs.init_real2sim(state_estimator, franka_arm)

                for step in range(episode_length):
                    start_time = time.perf_counter()

                    # Get action
                    next_obs = next_obs.to(self.device)
                    if self.args.debug_act and act_replay_buf is not None and step < len(act_replay_buf):
                        action = act_replay_buf[step].unsqueeze(0)
                    else:
                        action, _ = self.agent.get_action_and_value(next_obs, action_only=True)

                    # Record debug data
                    if self.args.debug_obs:
                        if self.args.use_sim_pure:
                            sim_robot_dict["obs"].append(next_obs[0].cpu().numpy())
                            if self.args.use_fk_replay:
                                real_robot_dict["obs"].append(infos_real.get("fk_obs", next_obs)[0].cpu().numpy())
                        else:
                            real_robot_dict["obs"].append(next_obs[0].cpu().numpy())

                    if self.args.debug_act:
                        real_robot_dict["action"].append(action[0].cpu().numpy())

                    # Execute action
                    if self.args.use_sim_pure:
                        next_obs_dict, reward, done, infos_real = self.envs.step(action)
                        next_obs = next_obs_dict["obs"]

                        if self.args.use_fk_replay and not self.args.not_move:
                            ctrl_cmd = infos_real["fk_replay_cmd"]
                            franka_arm.send_command(ctrl_cmd, cmd="fk_replay")

                        remaining_sleep(start_time, self.envs.ctrl_dt)

                        if self.args.use_fk_replay:
                            next_real_obs, _ = self.envs.compute_observations_real(state_estimator, franka_arm)
                            infos_real["fk_obs"] = next_real_obs
                    else:
                        ctrl_cmd = self.envs.pre_physics_step_real(action)
                        if not self.args.not_move:
                            franka_arm.send_command(ctrl_cmd, cmd=self.args.control_type)

                        remaining_sleep(start_time, self.envs.ctrl_dt)
                        next_obs, infos_real = self.envs.compute_observations_real(state_estimator, franka_arm)
                        reward, done, success = self.envs.compute_reward_real()

                    # Terminate if done
                    if done[0] == 1:
                        if self.args.use_sim_pure:
                            if self.args.use_fk_replay:
                                print(f"Fk_replay break")
                                break
                            self.envs.reset_idx(done.nonzero(as_tuple=False).squeeze(-1))
                            self.envs.init_real2sim(state_estimator, franka_arm)
                        else:
                            break

                    # Visualization
                    infos_real['image'] = demo_recorder.last_frame if demo_recorder.is_recording else None
                    self.draw_misc(next_obs, infos_real, done=done)
                    msg_send_count += 1
                    print(f"Sending msg count: {msg_send_count}")

            except Exception:
                traceback.print_exc()

            finally:
                demo_recorder.stop()
                state_estimator.stop()
                franka_arm.stop()
                print(f"Total real time: {time.perf_counter() - eps_start_time} seconds")
                self._save_debug_data(sim_robot_dict, real_robot_dict)


    def _save_debug_data(self, sim_robot_dict, real_robot_dict):
        """Save debug data to files."""
        cal_dir = "cal_results"

        if self.args.debug_obs:
            debug_obs_dir = os.path.join(cal_dir, "debug_obs")
            os.makedirs(debug_obs_dir, exist_ok=True)

            if self.args.use_sim_pure:
                obs_array = np.stack(sim_robot_dict["obs"], axis=0)
                np.save(os.path.join(debug_obs_dir, "sim_obs_record.npy"), obs_array)
                if self.args.use_fk_replay:
                    obs_array = np.stack(real_robot_dict["obs"], axis=0)
                    np.save(os.path.join(debug_obs_dir, "real_obs_record.npy"), obs_array)
            else:
                obs_array = np.stack(real_robot_dict["obs"], axis=0)
                np.save(os.path.join(debug_obs_dir, "real_obs_record.npy"), obs_array)

        if self.args.debug_act and real_robot_dict["action"]:
            debug_act_dir = os.path.join(cal_dir, "debug_act")
            os.makedirs(debug_act_dir, exist_ok=True)
            act_array = np.stack(real_robot_dict["action"], axis=0)
            file_name = "sim_act_record.npy" if (self.args.use_sim_pure and not self.args.use_fk_replay) else "real_act_record.npy"
            np.save(os.path.join(debug_act_dir, file_name), act_array)
