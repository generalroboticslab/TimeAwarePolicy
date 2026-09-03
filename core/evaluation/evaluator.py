import numpy as np
import os
import random
import time

import isaacgym
from envs import isaacgymenvs
import torch

from core.agents.agent import get_agent
from core.common.tensor import to_numpy
from core.evaluation.keyboard import KeyboardController
from core.evaluation.metrics import EvaluationMetricsMixin


class PolicyEvaluator(EvaluationMetricsMixin):
    """Evaluate trained policies in vectorized simulation.

    Project-specific metric and real-robot mixins are composed by the public
    project entrypoint rather than imported into this reusable module.
    """
    
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
        self.tensor_dtype = torch.float32
        
        # Setup
        self._setup_seeding()
        self._setup_environment()
        self._setup_agent()
        self._setup_visualizers()
        self._setup_metrics()
    
    
    def _setup_seeding(self):
        """Set random seeds for reproducibility."""
        random.seed(self.args.seed)
        np.random.seed(self.args.seed)
        torch.manual_seed(self.args.seed)
        torch.cuda.manual_seed_all(self.args.seed)
        torch.backends.cudnn.deterministic = self.args.torch_deterministic
    
    
    def _setup_environment(self):
        """Initialize Isaac Gym environment."""
        self.envs = isaacgymenvs.make(
            seed=self.args.seed,
            task=self.args.task_name,
            num_envs=self.args.num_envs,
            sim_device=self.args.sim_device,
            rl_device=self.args.sim_device,
            graphics_device_id=self.args.graphics_device_id,
            headless=self.args.graphics_device_id == -1,
            force_render=self.args.rendering,
            custom_args=self.args
        )
    
    
    def _setup_agent(self):
        """Initialize agent and load checkpoint."""
        self.agent = None
        if not self.args.random_policy and not self.args.heuristic_policy:
            self.agent = get_agent(self.envs, self.args, self.device)
            self.agent.load_checkpoint(
                self.args.checkpoint_path,
                evaluate=True,
                map_location=self.device,
            )
            self.agent.deterministic = self.args.deterministic
    
    
    def _setup_visualizers(self):
        """Initialize visualizers and controllers."""
        self.valVis = None
        self.robotVis = None
        
        if self.args.draw_time:
            from core.evaluation.visualization import ValueVisualizer

            self.valVis = ValueVisualizer(self.agent, xlim=(-0.2, 3.), ylim=(2, 5.5), vflip=self.args.time2end)
        elif self.args.draw_scevel_val:
            from core.evaluation.visualization import ValueVisualizer

            self.valVis = ValueVisualizer(self.agent, xlim=(-0.05, 0.5), ylim=(4.25, 5.5))
        
        if [self.args.draw_torque, self.args.draw_pos, self.args.draw_vel, self.args.draw_acc].count(True) > 0:
            from core.evaluation.visualization import RerunVis

            self.robotVis = RerunVis(dt=self.envs.ctrl_dt, num_joints=self.envs.num_franka_dofs)
        elif self.args.draw_scevel:
            from core.evaluation.visualization import RerunVis

            init_time_ratio = self.args.goal_speed if self.args.goal_speed is not None else 1.
            self.robotVis = RerunVis(
                dt=self.envs.ctrl_dt,
                num_joints=self.envs.num_franka_dofs,
                init_time_ratio=init_time_ratio,
                timeaware_layout=True,
                simple_layout=self.args.simple_layout,
                sliding_window=5
            )
            if self.args.keyboard_ctrl:
                keyboard_ctrl = KeyboardController(self.envs, start_value=init_time_ratio, start_config=self.args.specific_idx)
                keyboard_ctrl.start()
                self.robotVis.keyboard_ctrl = keyboard_ctrl
    
    
    def _setup_metrics(self):
        """Initialize metric tracking."""
        buffer_len = self.args.target_episodes
        if self.args.target_success_eps is not None or self.args.record_init_configs:
            self.args.target_episodes = int(3e6)
            buffer_len = self.args.target_success_eps + self.args.num_envs
        
        self.step_metrics = {
            "eps_r": torch.zeros((self.args.num_envs,), dtype=self.tensor_dtype).to(self.device),
        }
        self.initialize_paired_stage_metrics()
        if self.args.constraint_cost_eval:
            self.step_metrics["eps_instability_cost"] = torch.zeros(
                (self.args.num_envs,), dtype=self.tensor_dtype, device=self.device
            )
        
        self.eps_metrics = {
            key: torch.zeros((buffer_len,), dtype=self.tensor_dtype).to(self.device)
            for key in self.step_metrics.keys()
        }
        
        self.eps_metrics.update({
            "eps_time_p": torch.zeros((buffer_len,), dtype=self.tensor_dtype).to(self.device),
            "eps_time": torch.zeros((buffer_len,), dtype=self.tensor_dtype).to(self.device),
            "eps_time_goal": torch.zeros((buffer_len,), dtype=self.tensor_dtype).to(self.device),
            "eps_max_inst": torch.zeros((buffer_len,), dtype=self.tensor_dtype).to(self.device),
            "eps_stable_max_inst": torch.zeros((buffer_len,), dtype=self.tensor_dtype).to(self.device),
            "eps_lim_inst": torch.zeros((buffer_len,), dtype=self.tensor_dtype).to(self.device),
            "eps_sum_inst": torch.zeros((buffer_len,), dtype=self.tensor_dtype).to(self.device),
            "eps_success": torch.zeros((buffer_len,), dtype=self.tensor_dtype).to(self.device),
            "interaction_time": torch.zeros((buffer_len,), dtype=self.tensor_dtype).to(self.device),
        })
        
        self.eps_metrics_avg = {key: 0. for key in self.eps_metrics.keys()}
        self.eps_metrics_std = {key: 0. for key in self.eps_metrics.keys()}
        
        self.speed_and_time_dict = {
            "time_ratio": [],
            "time_used": [],
            "time_goal": [],
            "time_mismatch": [],
            "max_inst": [],
            "stable_max_inst": [],
            "sum_inst": [],
            "instability_cost": [],
            "thred_inst": [],
            "success_rate": [],
            "interaction_time": [],
        }
    
    
    def reset_obs_done(self):
        """Reset observation and done flags."""
        next_obs_dict = self.envs.reset()
        next_obs = torch.Tensor(next_obs_dict["obs"]).to(self.device)
        next_done = torch.zeros(self.args.num_envs).to(self.device)
        next_lstm_state = None
        
        if self.args.use_lstm:
            next_lstm_state = (
                torch.zeros(self.agent.crt_lstm.num_layers, self.args.num_envs, self.agent.crt_lstm.hidden_size).to(self.device),
                torch.zeros(self.agent.crt_lstm.num_layers, self.args.num_envs, self.agent.crt_lstm.hidden_size).to(self.device),
                torch.zeros(self.agent.act_lstm.num_layers, self.args.num_envs, self.agent.act_lstm.hidden_size).to(self.device),
                torch.zeros(self.agent.act_lstm.num_layers, self.args.num_envs, self.agent.act_lstm.hidden_size).to(self.device),
            )
        
        return next_obs, next_done, next_lstm_state
    
    
    def draw_misc(self, next_obs, infos, next_lstm_state=None, done=None):
        """Draw miscellaneous visualizations."""
        if self.args.draw_time:
            s_goal = to_numpy(infos["eps_time_goal"][0]) if not self.args.time2end else 0
            self.valVis.draw_value(
                next_obs.clone(),
                scur_idx=infos["tobs_idx"],
                step=0.1,
                lstm_state=next_lstm_state,
                done=done,
                s_goal=s_goal,
                pause=0.01
            )
        
        if self.args.draw_scevel_val:
            s_goal = to_numpy(infos["scene_linvel_lim"])
            self.valVis.draw_value(
                next_obs.clone(),
                scur_idx=infos["vobs_idx"],
                step=0.02,
                lstm_state=next_lstm_state,
                done=done,
                s_goal=s_goal,
                pause=0.1,
                s_name="Vel"
            )
        
        if self.args.draw_torque:
            self.robotVis.log_joint(
                values=infos["joint_torqs"],
                uppers=infos["franka"]["effort_limits"],
                lowers=-infos["franka"]["effort_limits"]
            )
        elif self.args.draw_pos:
            self.robotVis.log_joint(
                values=infos["joint_poss"],
                extras=infos["joint_tgt_q"],
            )
        elif self.args.draw_vel:
            self.robotVis.log_joint(
                values=infos["joint_vels"],
                uppers=infos["joint_velocity_limits"],
                lowers=-infos["joint_velocity_limits"],
            )
        elif self.args.draw_acc:
            self.robotVis.log_joint(values=infos["joint_accs"])
        elif self.args.draw_scevel:
            self.robotVis.log_timeaware_step(
                img=infos.get("image", None),
                observed_tte=max(infos["observed_time2end"].item(), 0.),
                real_tte=infos["real_time2end"],
                cur_time_ratio=infos["time_ratio"].item(),
                instability=infos["scene_linvel"].item(),
                instability_lim=infos["scene_linvel_lim"].item(),
                done=done[0]
            )
    
    
    def scan_all_time(self, next_obs):
        """Scan all time steps for analysis."""
        num_time_steps = 10
        num_ratio_steps = 9
        proposed_time = torch.linspace(self.envs.time2end_init[0], 0., steps=num_time_steps).to(self.device)
        proposed_time_ratio = torch.linspace(0.2, 1., steps=num_ratio_steps).to(self.device)
        
        time_obs = next_obs.repeat(num_time_steps, 1)
        time_obs[:, -1] = proposed_time
        sr_obs = next_obs.repeat(num_ratio_steps, 1)
        sr_obs[:, -2] = proposed_time_ratio
        pure_obs = self.envs.obs_dict["states"][0].to(self.device)
        
        time_vary_actions, time_vary_grad_obs = self.agent.logprob_saliency(time_obs)
        time_vary_actions = self.envs.convert_actions(time_vary_actions)
        
        sr_vary_actions, sr_vary_grad_obs = self.agent.logprob_saliency(sr_obs)
        sr_vary_actions = self.envs.convert_actions(sr_vary_actions)
        
        return pure_obs, proposed_time, proposed_time_ratio, time_vary_actions, sr_vary_actions, time_vary_grad_obs, sr_vary_grad_obs
    
    
    def draw_scan_time_res(self, next_obs, num_episodes, fig3d=None, ax3d=None, cbar3d=None, fig2d=None, ax2d=None, cbar2d=None):
        """Draw scan time results."""
        from core.evaluation.visualization import show_heatmap
        from projects.TimeAwarePolicy.evaluation.scene_visualization import (
            visualize_scene_3d,
        )

        pure_obs, proposed_time, proposed_time_ratio, time_vary_actions, sr_vary_actions, time_vary_grad_obs, sr_vary_grad_obs = self.scan_all_time(next_obs)
        
        step_dir = os.path.join(self.args.scan_time_save_dir, f"{num_episodes}_step_{str(self.envs.policy_steps[0].item())}")
        os.makedirs(step_dir, exist_ok=True)
        
        save_path = os.path.join(step_dir, "time_var_3D.pdf")
        fig3d, ax3d, cbar3d = visualize_scene_3d(
            pure_obs.cpu().numpy(),
            time_vary_actions.cpu().numpy(),
            proposed_time.cpu().numpy(),
            save_path=save_path,
            fig=fig3d, ax=ax3d, cbar=cbar3d
        )
        
        save_path = os.path.join(step_dir, "sr_var_3D.pdf")
        fig3d, ax3d, cbar3d = visualize_scene_3d(
            pure_obs.cpu().numpy(),
            sr_vary_actions.cpu().numpy(),
            proposed_time_ratio.cpu().numpy(),
            save_path=save_path,
            cmap_name="viridis_r",
            revert_y=True,
            fig=fig3d, ax=ax3d, cbar=cbar3d
        )
        
        save_path = os.path.join(step_dir, "time_var_heatmap.pdf")
        fig2d, ax2d, cbar2d = show_heatmap(
            time_vary_grad_obs.cpu().numpy(),
            yticklabels=proposed_time.cpu().numpy(),
            save_path=save_path,
            fig=fig2d, ax=ax2d, cbar=cbar2d
        )
        
        save_path = os.path.join(step_dir, "sr_var_heatmap.pdf")
        fig2d, ax2d, cbar2d = show_heatmap(
            sr_vary_grad_obs.cpu().numpy(),
            yticklabels=proposed_time_ratio.cpu().numpy(),
            save_path=save_path,
            fig=fig2d, ax=ax2d, cbar=cbar2d
        )
        
        return fig3d, ax3d, cbar3d, fig2d, ax2d, cbar2d
    
    
    def evaluate_simulation(self):
        """Run simulation-based evaluation."""
        if self.args.task_name == "FrankaCubeStack":
            dynamic_change_lst = self.args.disturbance_v_lst
            dynamic_change_name = "disturbance_v"
        elif self.args.task_name == "FrankaGmPour":
            dynamic_change_lst = self.args.num_gms_lst
            dynamic_change_name = "num_gms"
        elif self.args.task_name == "FrankaCabinet":
            dynamic_change_lst = self.args.friction_mul_lst
            dynamic_change_name = "friction_mul"
        
        self.speed_and_time_dict[dynamic_change_name] = []
        
        if self.args.scan_time:
            fig3d = None
            ax3d = None
            cbar3d = None
            fig2d = None
            ax2d = None
            cbar2d = None
        
        with torch.no_grad():
            for i in range(max(1, len(self.args.goal_speed_lst))):
                for j in range(max(1, len(dynamic_change_lst))):
                    cur_goal_speed = self.envs.goal_speed = self.args.goal_speed_lst[i]
                    
                    if self.args.task_name == "FrankaCubeStack":
                        cur_dynamic_v = self.envs.disturbance_v = self.args.disturbance_v_lst[j]
                    elif self.args.task_name == "FrankaGmPour":
                        cur_dynamic_v = self.envs.num_gms = self.args.num_gms_lst[j]
                    elif self.args.task_name == "FrankaCabinet":
                        cur_dynamic_v = self.envs.friction_mul = self.args.friction_mul_lst[j]
                    
                    self.envs.reset_all()
                    next_obs, next_done, next_lstm_state = self.reset_obs_done()
                    
                    temp_num_episodes = 0
                    warmup_end = False if (self.args.warmup_episodes > 0) else True
                    self.agent.deterministic = self.args.deterministic
                    
                    # Reset buffers
                    num_episodes = 0
                    num_success_eps = 0
                    valid_env_ids = torch.arange(self.args.num_envs, device=self.device)
                    
                    for key in self.step_metrics.keys():
                        self.step_metrics[key][:] = 0.
                    for key in self.eps_metrics.keys():
                        self.eps_metrics[key][:] = 0.
                    
                    print(f"Start Evaluating: {self.args.target_episodes} Trials | "
                          f"Goal Speed: {cur_goal_speed} | {dynamic_change_name}: {cur_dynamic_v} | "
                          f"{self.args.target_success_eps} Success Trials Required")
                    if not warmup_end:
                        print(f"Warmup Episodes: {temp_num_episodes}/{self.args.warmup_episodes}")
                    
                    start_time = time.perf_counter()
                    
                    while num_episodes < self.args.target_episodes:
                        # Get action
                        if self.args.random_policy or self.args.heuristic_policy:
                            action = torch.rand((self.args.num_envs, self.envs.num_actions), device=self.device)
                        else:
                            if self.args.use_lstm:
                                action, probs, next_lstm_state = self.agent.get_action_and_value(
                                    next_obs, next_lstm_state, next_done, action_only=True
                                )
                            else:
                                action, probs = self.agent.get_action_and_value(next_obs, action_only=True)
                        
                        # Step environment
                        next_obs_dict, reward, done, infos = self.envs.step(action)
                        next_obs = next_obs_dict["obs"].to(self.device)
                        next_done = done.to(self.device)
                        rewards = reward.to(self.device).view(-1)
                        self.step_metrics['eps_r'] += rewards
                        if self.args.constraint_cost_eval:
                            # Cost channel 1 is c_t^inst = max(p_t - p_max * tr, 0).
                            self.step_metrics["eps_instability_cost"] += infos["cost"][:, 1]
                        
                        # Rendering and visualization
                        if warmup_end:
                            if self.args.rendering:
                                self.draw_misc(next_obs, infos, next_lstm_state, next_done)
                            
                            if self.args.scan_time:
                                fig3d, ax3d, cbar3d, fig2d, ax2d, cbar2d = self.draw_scan_time_res(
                                    next_obs, num_episodes, fig3d, ax3d, cbar3d, fig2d, ax2d, cbar2d
                                )
                        
                        # Handle episode termination
                        terminal_index = done == 1
                        terminal_nums = terminal_index.sum().item()
                        
                        if terminal_nums > 0:
                            if (temp_num_episodes < self.args.warmup_episodes) and (not warmup_end):
                                temp_num_episodes += terminal_nums
                                if temp_num_episodes >= self.args.warmup_episodes:
                                    warmup_end = True
                                    print(f"End Warmup Episodes: {temp_num_episodes}/{self.args.warmup_episodes}")
                                else:
                                    for key in self.step_metrics.keys():
                                        self.step_metrics[key][:] = 0.
                                    continue
                            
                            if self.args.strict_eval:
                                terminal_index = (valid_env_ids != -1) & terminal_index
                                terminal_nums = terminal_index.sum().item()
                                valid_env_ids[terminal_index] = -1
                                if terminal_nums == 0:
                                    continue
                            
                            num_episodes += terminal_nums
                            num_success_eps += self.update_episode_metrics(terminal_index, infos)
                            num_eps_recorded = infos.get("num_eps_recorded", 0)
                            
                            print_info = f"Episodes: {num_episodes} | Total Success: {num_success_eps}"
                            if self.args.record_init_configs:
                                print_info += f" | Recorded Eps: {num_eps_recorded}/{self.args.target_success_eps}"
                            if not self.args.keyboard_ctrl:
                                print(print_info)
                            
                            # Check termination conditions
                            if self.args.record_init_configs:
                                if infos.get("update_done", False):
                                    break
                                elif not self.args.update_configs and num_eps_recorded >= self.args.target_success_eps:
                                    break
                            else:
                                if self.args.target_success_eps is not None and num_success_eps >= self.args.target_success_eps:
                                    break
                            
                            if self.args.strict_eval and (valid_env_ids == -1).all():
                                print(f"All envs have been evaluated, break the loop")
                                break
                    
                    # Compute and save results
                    self.compute_average_metrics(num_episodes, num_success_eps)
                    self.update_speed_time_dict(cur_goal_speed, cur_dynamic_v)
                    machine_time = time.perf_counter() - start_time
                    
                    print(f"{self.args.target_episodes} Episodes | {num_eps_recorded} Recorded Episodes "
                          f"| Success Rate: {self.eps_metrics_avg['eps_success'] * 100:.3f}% "
                          f"| Avg Reward: {self.eps_metrics_avg['eps_r']:.3f} "
                          f"| Avg Time Used: {self.eps_metrics_avg['eps_time']:.3f} "
                          f"| Avg Time Goal: {self.eps_metrics_avg['eps_time_goal']:.3f} "
                          f"| Avg Time Mismatch: {self.eps_metrics_avg['eps_time_p']:.3f} "
                          f"| Avg Sum Eps Instability: {self.eps_metrics_avg['eps_sum_inst']:.3f} +- {self.eps_metrics_std['eps_sum_inst']:.3f} "
                          + (f"| Avg Eps Instability Cost: {self.eps_metrics_avg['eps_instability_cost']:.3f} +- {self.eps_metrics_std['eps_instability_cost']:.3f} "
                             if self.args.constraint_cost_eval else "") +
                          f"| Avg Max Eps Instability: {self.eps_metrics_avg['eps_max_inst']:.3f} "
                          f"| Target Thred Instability: {infos['scene_linvel_lim'].item():.3f} "
                          f"| Avg Manipulation Time: {self.eps_metrics_avg['interaction_time']:.3f} "
                          f"| Time: {machine_time:.3f} | Num of Env: {self.args.num_envs}\n")
                    
                    self.save_results(num_episodes, machine_time, num_eps_recorded, infos)
    
    
    def run(self):
        """Main evaluation entry point."""
        if self.args.real_robot:
            self.evaluate_real_robot()
        else:
            self.evaluate_simulation()
        
        print('Process Over')
