import os
import random
import time
import wandb
from math import ceil
import numpy as np
import isaacgym
import torch
import torch.nn as nn
import torch.optim as optim
from tabulate import tabulate
import threadpoolctl as tpc
import multiprocessing

from envs import isaacgymenvs
from core.agents.agent import (
    get_agent,
    get_meta_agent,
    load_checkpoint,
    save_checkpoint,
)
from core.agents.normalization import NormalizeReward
from core.agents.utils import (
    AdaptiveScheduler,
    LinearScheduler,
)
from core.common.time import convert_time
from core.training.checkpointing import CheckpointingMixin
from core.training.curriculum import CurriculumMixin
from core.training.logging import TrainingLoggingMixin
from core.training.algorithms.policy_updates import PolicyUpdateMixin
from core.training.rollout import RolloutMixin


class PolicyTrainer(
    PolicyUpdateMixin,
    CurriculumMixin,
    TrainingLoggingMixin,
    CheckpointingMixin,
    RolloutMixin,
):
    """Train PPO and constrained-policy variants for vectorized tasks."""
    
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
        self.tensor_dtype = torch.float32
        
        # Initialize components
        self._setup_seeding()
        self._setup_environment()
        self._setup_agent()
        self._setup_optimizer()
        self._setup_normalizers()
        self._setup_storage()
        self._setup_tracking()
        self._setup_wandb()
        
        # Compute batch sizes
        self.args.batch_size = int(self.args.num_envs * self.args.num_steps)
        self.args.minibatch_size = self.args.batch_size if self.args.minibatch_size is None else self.args.minibatch_size
        self.args.num_minibatches = max(ceil(self.args.batch_size // self.args.minibatch_size), 1)
        
        self._print_configuration()
    
    
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
    
    
    def _setup_seeding(self):
        """Set random seeds for reproducibility."""
        random.seed(self.args.seed)
        np.random.seed(self.args.seed)
        torch.manual_seed(self.args.seed)
        torch.cuda.manual_seed_all(self.args.seed)
        torch.backends.cudnn.deterministic = self.args.torch_deterministic
    
    
    def _setup_agent(self):
        """Initialize agent and load checkpoint if provided."""
        self.agent = self.rollout_agent = get_agent(self.envs, self.args, self.device)
        
        if self.args.checkpoint is not None:
            checkpoint_folder = os.path.join(self.args.train_res_dir, self.args.checkpoint, "checkpoints")
            self.args.checkpoint_path = os.path.join(checkpoint_folder, f"eps_{self.args.index_episode}")
            assert os.path.exists(self.args.checkpoint_path)
            self.rollout_agent.load_checkpoint(self.args.checkpoint_path, map_location=self.device, reset_critic=self.args.reset_critic)
            
            if self.args.stu_train:
                self.agent = get_agent(self.envs, self.args, self.device)
                self.rollout_agent.set_mode('eval')
        
        if self.args.meta_rl:
            pl_agent = self.agent
            pl_agent.set_mode('eval')
            self.envs.pl_agent = pl_agent
            self.agent = get_meta_agent(self.envs, self.args, self.device)
        
        self.agent.set_mode('train')
    
    
    def _setup_optimizer(self):
        """Initialize optimizer and learning rate scheduler."""
        if self.args.cmdp_method == "cpo":
            critic_parameters = []
            for name, parameter in self.agent.named_parameters():
                if name.startswith(("critic.", "critic_t.", "critic_inst.")):
                    critic_parameters.append(parameter)
            self.optimizer = optim.Adam(critic_parameters, lr=self.args.lr, eps=1e-5)
        else:
            self.optimizer = optim.Adam(self.agent.parameters(), lr=self.args.lr, eps=1e-5)
        
        if self.args.scheduler == 'adapt':
            self.lr_scheduler = AdaptiveScheduler(kl_threshold=1.6e-2)
        elif self.args.scheduler == 'linear':
            self.lr_scheduler = LinearScheduler(start_lr=self.args.lr, max_steps=self.args.total_timesteps)
        else:
            raise NotImplementedError(f"Scheduler {self.args.scheduler} is not implemented")
    
    
    def _setup_normalizers(self):
        """Initialize reward and cost normalizers."""
        self.reward_normalizer = None
        if self.args.norm_rew:
            self.reward_normalizer = NormalizeReward(self.args.num_envs, device=self.device)
            if self.args.checkpoint is not None and not self.args.reset_critic:
                checkpoint_folder = os.path.join(self.args.train_res_dir, self.args.checkpoint, "checkpoints")
                rew_ckpt_path = os.path.join(checkpoint_folder, f"rew_norm_eps_{self.args.index_episode}")
                if os.path.exists(rew_ckpt_path):
                    self.reward_normalizer = load_checkpoint(self.reward_normalizer, rew_ckpt_path, evaluate=False, map_location=self.device)
                else:
                    print(f"WARN: Reward normalizer checkpoint {rew_ckpt_path} does not exist!")
        
        self.cost_normalizer = None
        if self.args.use_cost and self.args.norm_cost:
            c_gamma = torch.tensor(self.args.c_gamma, dtype=self.tensor_dtype, device=self.device).view(1, -1)
            self.cost_normalizer = NormalizeReward(self.args.num_envs, gamma=c_gamma, insize=self.args.num_cost, device=self.device)
    
    
    def _setup_storage(self):
        """Initialize storage buffers for training data."""
        state_shape = self.envs.state_space.shape
        obs_shape = self.envs.obs_space.shape
        act_shape = self.envs.act_space.shape if not self.args.meta_rl else (2,)
        
        self.obs = torch.zeros((self.args.num_steps, self.args.num_envs) + obs_shape, dtype=self.tensor_dtype, device=self.device)
        self.states = torch.zeros((self.args.num_steps, self.args.num_envs) + state_shape, dtype=self.tensor_dtype, device=self.device)
        self.actions = torch.zeros((self.args.num_steps, self.args.num_envs) + act_shape, dtype=self.tensor_dtype, device=self.device)
        self.logprobs = torch.zeros((self.args.num_steps, self.args.num_envs), dtype=self.tensor_dtype, device=self.device)
        self.rewards = torch.zeros((self.args.num_steps, self.args.num_envs), dtype=self.tensor_dtype, device=self.device)
        self.dones = torch.zeros((self.args.num_steps, self.args.num_envs), dtype=self.tensor_dtype, device=self.device)
        self.timeouts = torch.zeros((self.args.num_steps, self.args.num_envs), dtype=self.tensor_dtype, device=self.device)
        self.values = torch.zeros((self.args.num_steps, self.args.num_envs), dtype=self.tensor_dtype, device=self.device)
        
        if self.args.use_cost:
            self.costs = torch.zeros((self.args.num_steps, self.args.num_envs, self.args.num_cost), dtype=self.tensor_dtype, device=self.device)
            self.values_c = torch.zeros((self.args.num_steps, self.args.num_envs, self.args.num_cost), dtype=self.tensor_dtype, device=self.device)
            self.c_gamma = torch.tensor(self.args.c_gamma, dtype=self.tensor_dtype, device=self.device).view(1, -1)
            self.c_scale = torch.tensor(self.args.c_scale, dtype=self.tensor_dtype, device=self.device).view(1, -1)
            if self.args.cmdp_method == "ppo_lagrangian":
                self.lagrange_multipliers = torch.full(
                    (self.args.num_cost,),
                    self.args.lagrangian_init,
                    dtype=self.tensor_dtype,
                    device=self.device,
                )
        
        # Reset environment
        next_obs_dict = self.envs.reset()
        self.next_obs = torch.Tensor(next_obs_dict["obs"]).to(self.device)
        self.next_state = torch.Tensor(next_obs_dict["states"]).to(self.device)
        self.next_done = torch.zeros(self.args.num_envs, device=self.device)
        self.next_timeout = torch.zeros(self.args.num_envs, device=self.device)
        
        if self.args.stu_train:
            self.tea_next_obs = torch.Tensor(next_obs_dict["fixed_obs"]).to(self.device)
            self.tea_next_state = torch.Tensor(next_obs_dict["fixed_state"]).to(self.device)
        
        if self.args.use_lstm:
            self.next_lstm_state = (
                torch.zeros(self.rollout_agent.crt_lstm.num_layers, self.args.num_envs, self.rollout_agent.crt_lstm.hidden_size, dtype=self.tensor_dtype, device=self.device),
                torch.zeros(self.rollout_agent.crt_lstm.num_layers, self.args.num_envs, self.rollout_agent.crt_lstm.hidden_size, dtype=self.tensor_dtype, device=self.device),
                torch.zeros(self.rollout_agent.act_lstm.num_layers, self.args.num_envs, self.rollout_agent.act_lstm.hidden_size, dtype=self.tensor_dtype, device=self.device),
                torch.zeros(self.rollout_agent.act_lstm.num_layers, self.args.num_envs, self.rollout_agent.act_lstm.hidden_size, dtype=self.tensor_dtype, device=self.device),
            )
    
    
    def _setup_tracking(self):
        """Initialize tracking variables for training metrics."""
        # Global counters
        self.global_update_iter = 0
        self.attempted_update_iter = 0
        self.skipped_update_iter = 0
        self.global_step = 0
        self.global_episodes = 0
        self.reward_update_iters = 0
        self.reward_steps = 0
        self.reward_episodes = 0
        
        # Episode statistics
        self.step_r_store = {
            "eps_r": torch.zeros((self.args.num_envs,), dtype=self.tensor_dtype).to(self.device),
            "eps_scenevel_p": torch.zeros((self.args.num_envs,), dtype=self.tensor_dtype).to(self.device),
            "eps_sceneacc_p": torch.zeros((self.args.num_envs,), dtype=self.tensor_dtype).to(self.device),
            "eps_act_p": torch.zeros((self.args.num_envs,), dtype=self.tensor_dtype).to(self.device),
            "eps_c": torch.zeros((self.args.num_envs,), dtype=self.tensor_dtype).to(self.device),
        }
        
        self.eps_r_store = {
            "success": torch.zeros((self.args.running_len,), dtype=self.tensor_dtype).to(self.device),
            "eps_time": torch.zeros((self.args.running_len,), dtype=self.tensor_dtype).to(self.device),
            "eps_horizon": torch.zeros((self.args.running_len,), dtype=self.tensor_dtype).to(self.device),
            "eps_time_p": torch.zeros((self.args.running_len,), dtype=self.tensor_dtype).to(self.device),
            "eps_max_scevel": torch.zeros((self.args.running_len,), dtype=self.tensor_dtype).to(self.device),
        }
        
        for key in self.step_r_store.keys():
            self.eps_r_store[key] = torch.zeros((self.args.running_len,), dtype=self.tensor_dtype).to(self.device)
        
        self.eps_r_avg = {key: 0 for key in self.eps_r_store.keys()}
        
        self.success_record_keys = ["eps_time", "eps_time_p"]
        for key in self.success_record_keys:
            if key not in self.eps_r_store:
                raise ValueError(f"Success only recorded key '{key}' is not in eps_r_store!")
        
        # Best metrics
        self.cur_checkpoint_score = -torch.inf
        self.cur_success_rate = 0.
        self.cur_eps_time = 0.
        self.cur_loss = torch.inf
        self.best_checkpoint_score = -torch.inf
        self.best_success_rate = 0.
        self.max_eps_time = 0.
        self.best_loss = torch.inf
        
        # Curriculum
        self.curri_episodes = 0
        self.curri_steps = 0
        self.success_episodes = 0
        self.curri_update_iters = 0
        self.curriculum_above = 0
        self.curriculum_below = 0
        self.curri_ratio = self.args.init_curri_ratio
        self.ready_to_record = False
        self.avg_buffer_reset = True
        
        # Curriculum values
        self.cur_ent = self.args.ent_coef[0]
        self.envs.cfg['r_epstime_scale'] = self.args.epstimeRewardScale[0]
        self.envs.cfg['r_scene_vel_scale'] = self.args.scevelRewardScale[0]
        
        # Metadata
        self.meta_data = {"quality_candidates": {}, "training_info": {}}
        self.quality_candidates = self.meta_data["quality_candidates"]
        self.training_info = self.meta_data["training_info"]
        self.quality_candidate_start_update = None
        self.quality_candidate_last_update = None
        
        self.start_time = time.time()
    
    
    def _setup_wandb(self):
        """Initialize Weights & Biases logging."""
        config = dict(
            Name=self.args.env_name,
            algorithm=self.args.cmdp_method if self.args.use_cost else 'ppo',
            num_envs=self.args.num_envs,
            lr=self.args.lr,
            gamma=self.args.gamma,
            alpha=self.args.ent_coef,
            deterministic=self.args.deterministic,
            sequence_len=self.args.sequence_len,
            random_policy=self.args.random_policy,
        )
        
        if self.args.saving and self.args.wandb:
            wandb_kwargs = {
                'project': self.args.wandb_project or self.args.env_name,
                'config': config,
                'name': self.args.final_name,
            }
            if self.args.wandb_entity:
                wandb_kwargs['entity'] = self.args.wandb_entity
            wandb.init(**wandb_kwargs)
        else:
            wandb.init(mode="disabled")
    
    
    def _print_configuration(self):
        """Print training configuration."""
        raw_obs_shape_data = [
            ["Summary", ""],
            ["Num Envs", self.envs.num_envs],
            ["Sequence Len", self.args.sequence_len],
            ["Observation Shape", self.envs.observation_space.shape],
            ["State Shape", self.envs.state_space.shape],
            ["Action Shape", self.envs.action_space.shape],
        ]
        print(tabulate(raw_obs_shape_data, headers="firstrow", tablefmt="grid"))
        
        print(f"########### ATTENTION ###########\n"
              f"Uniform Name: {self.args.final_name}\n\n"
              f"Batch Size: {self.args.batch_size}, MiniBatchSize: {self.args.minibatch_size}, "
              f"Num Minibatches: {self.args.num_minibatches}, Num UpdateEpochs: {self.args.update_epochs}\n"
              f"#################################\n")
    
    
    def update_student_policy(self, initial_lstm_state=None):
        """Update student policy using behavior cloning (only in the embed temporal observation stage)."""
        obs_shape = self.envs.obs_space.shape
        state_shape = self.envs.state_space.shape
        act_shape = self.envs.act_space.shape if not self.args.meta_rl else (2,)
        
        b_obs = self.obs.reshape((-1,) + obs_shape)
        b_states = self.states.reshape((-1,) + state_shape)
        b_logprobs = self.logprobs.reshape(-1)
        b_actions = self.actions.reshape((-1,) + act_shape)
        b_values = self.values.reshape(-1)
        
        if self.args.use_lstm:
            envsperbatch = self.args.num_envs // self.args.num_minibatches
            envinds = np.arange(self.args.num_envs)
            flatinds = np.arange(self.args.batch_size).reshape(self.args.num_steps, self.args.num_envs)
            end_idx = self.args.num_envs
            step_num = envsperbatch
            b_dones = self.dones.reshape(-1)
        else:
            b_inds = np.arange(self.args.batch_size)
            end_idx = self.args.batch_size
            step_num = self.args.minibatch_size
        
        for epoch in range(self.args.update_epochs):
            if self.args.use_lstm:
                np.random.shuffle(envinds)
            else:
                np.random.shuffle(b_inds)
            
            for start in range(0, end_idx, step_num):
                end = start + step_num
                
                if self.args.use_lstm:
                    mbenvinds = envinds[start:end]
                    mb_inds = flatinds[:, mbenvinds].ravel()
                    _, mu, newlogprob, entropy, newvalue, _, _ = self.agent.get_action_and_value(
                        b_obs[mb_inds],
                        b_states[mb_inds],
                        (initial_lstm_state[0][:, mbenvinds], initial_lstm_state[1][:, mbenvinds],
                         initial_lstm_state[2][:, mbenvinds], initial_lstm_state[3][:, mbenvinds]),
                        b_dones[mb_inds],
                        b_actions[mb_inds],
                    )
                else:
                    mb_inds = b_inds[start:end]
                    _, mu, newlogprob, entropy, newvalue, _ = self.agent.get_action_and_value(
                        b_obs[mb_inds],
                        b_states[mb_inds],
                        b_actions[mb_inds]
                    )
                
                ratio_loss = 0.5 * ((newlogprob - b_logprobs[mb_inds]) ** 2).mean()
                newvalue = newvalue.view(-1)
                v_loss = 0.5 * ((newvalue - b_values[mb_inds]) ** 2).mean()
                loss = ratio_loss + v_loss * self.args.vf_coef
                
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.agent.critic.parameters(), self.args.max_grad_norm)
                nn.utils.clip_grad_norm_(self.agent.actor.parameters(), self.args.max_grad_norm)
                self.optimizer.step()
                
                if self.args.anneal_lr:
                    if self.args.scheduler == 'adapt':
                        new_lr = self.lr_scheduler.update(self.optimizer.param_groups[0]["lr"], 0)
                    else:
                        new_lr, _ = self.lr_scheduler.update(self.global_step)
                    self.optimizer.param_groups[0]["lr"] = new_lr
        
        self.cur_loss = loss.item()
        if self.cur_loss < self.best_loss:
            self.best_loss = self.cur_loss
            if self.args.saving:
                self.agent.save_checkpoint(folder_path=self.args.checkpoint_dir, suffix='best', reward_normalizer=self.reward_normalizer)
        
        return ratio_loss.item(), v_loss.item()
    
    
    def train(self):
        """Main training loop."""
        n_cpu_cores = multiprocessing.cpu_count()
        n_gpu_used = 1
        # Limit the number of threads used for training
        thread_limits = max(4, int(n_cpu_cores * n_gpu_used / self.args.num_envs))
        
        with tpc.threadpool_limits(limits=thread_limits):
            torch.cuda.empty_cache()

            num_updates = max(self.args.total_timesteps // self.args.batch_size, 1)
            
            for update in range(num_updates):
                start_time = time.perf_counter()
                self.attempted_update_iter = update + 1
                
                # Collect rollout
                initial_lstm_state = self.collect_rollout()
                
                # Log episode metrics
                self.log_episode_metrics()
                
                # Print status
                self.print_status(update, num_updates)
                
                # Skip training for random policy
                if self.args.random_policy:
                    continue
                
                # Student training
                if self.args.stu_train:
                    ratio_loss, v_loss = self.update_student_policy(initial_lstm_state)
                    self.print_student_status(update, num_updates, ratio_loss, v_loss)
                    self.global_update_iter += 1
                    self.curri_update_iters += 1
                    continue
                
                # Compute advantages
                returns, advantages, returns_c, advantages_c = self.compute_advantages(initial_lstm_state)
                
                # Update policy
                policy_diverged = self.update_policy(returns, advantages, returns_c, advantages_c, initial_lstm_state)
                
                if policy_diverged:
                    continue
                
                # Update counters and curriculum
                self.global_update_iter += 1
                self.curri_update_iters += 1
                self.save_quality_candidate()
                self.update_curriculum()
                
                # Save checkpoints
                self.save_checkpoints()

                if not self.args.quiet:
                    elapsed = time.time() - self.start_time
                    print(f"\nRunning Time: {convert_time(elapsed)}, "
                          f"Update Time: {time.perf_counter() - start_time:.2f}s for {self.args.update_epochs * self.args.num_minibatches} minibatch iterations, "
                          f"Global Steps: {self.global_step}, "
                          f"Update Iteration: {self.global_update_iter}")
            
            # Save final checkpoint
            if self.args.saving and not self.args.random_policy:
                self.agent.save_checkpoint(folder_path=self.args.checkpoint_dir, suffix='last')
                save_checkpoint(self.reward_normalizer, self.args.checkpoint_dir, ckpt_name="rew_norm_eps", suffix='last')
            
            print('\nProcess Over here')
            if hasattr(self.envs, 'close'):
                self.envs.close()
            wandb.finish()
