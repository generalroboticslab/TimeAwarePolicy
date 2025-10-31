import json
import os
import random
import time
import wandb
from math import ceil
from copy import deepcopy
import numpy as np
import isaacgym
import torch
import torch.nn as nn
import torch.optim as optim
from tabulate import tabulate
import threadpoolctl as tpc
import multiprocessing

from envs import isaacgymenvs
from model.agent import *
from model.utils import *
from tw_training_utils import *
from utils import *


class PPOTrainer:
    """RL Trainer."""
    
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
        self.tensor_dtype = torch.float32
        
        # Initialize components
        self._setup_environment()
        self._setup_seeding()
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
        self.args.graphics_device_id = 2 if self.args.rendering else -1
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
            checkpoint_folder = os.path.join(self.args.result_dir, self.args.checkpoint, "checkpoints")
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
                checkpoint_folder = os.path.join(self.args.result_dir, self.args.checkpoint, "checkpoints")
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
        self.values = torch.zeros((self.args.num_steps, self.args.num_envs), dtype=self.tensor_dtype, device=self.device)
        
        if self.args.use_cost:
            self.costs = torch.zeros((self.args.num_steps, self.args.num_envs, self.args.num_cost), dtype=self.tensor_dtype, device=self.device)
            self.values_c = torch.zeros((self.args.num_steps, self.args.num_envs, self.args.num_cost), dtype=self.tensor_dtype, device=self.device)
            self.c_gamma = torch.tensor(self.args.c_gamma, dtype=self.tensor_dtype, device=self.device).view(1, -1)
            self.c_scale = torch.tensor(self.args.c_scale, dtype=self.tensor_dtype, device=self.device).view(1, -1)
        
        # Reset environment
        next_obs_dict = self.envs.reset()
        self.next_obs = torch.Tensor(next_obs_dict["obs"]).to(self.device)
        self.next_state = torch.Tensor(next_obs_dict["states"]).to(self.device)
        self.next_done = torch.zeros(self.args.num_envs, device=self.device)
        
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
        
        self.success_record_keys = ["eps_time"]
        for key in self.success_record_keys:
            if key not in self.eps_r_store:
                raise ValueError(f"Success only recorded key '{key}' is not in eps_r_store!")
        
        # Best metrics
        self.cur_rew = -torch.inf
        self.cur_success_rate = 0.
        self.cur_eps_time = 0.
        self.cur_loss = torch.inf
        self.best_rew = -torch.inf
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
        self.meta_data = {"milestone": {}, "training_info": {}}
        self.milestone = self.meta_data["milestone"]
        self.training_info = self.meta_data["training_info"]
        
        self.start_time = time.time()
    
    
    def _setup_wandb(self):
        """Initialize Weights & Biases logging."""
        config = dict(
            Name=self.args.env_name,
            algorithm='PPO Continuous',
            num_envs=self.args.num_envs,
            lr=self.args.lr,
            gamma=self.args.gamma,
            alpha=self.args.ent_coef,
            deterministic=self.args.deterministic,
            sequence_len=self.args.sequence_len,
            random_policy=self.args.random_policy,
        )
        
        if self.args.saving and self.args.wandb:
            wandb.init(project=self.args.env_name, entity='jiayinsen', config=config, name=self.args.final_name)
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
    
    
    def collect_rollout(self):
        """Collect rollout data from environment."""
        if self.args.use_lstm:
            initial_lstm_state = [lstm_state.clone() for lstm_state in self.next_lstm_state]
        
        for step in range(self.args.num_steps):
            self.global_step += self.args.num_envs
            self.obs[step] = self.next_obs
            self.states[step] = self.next_state
            self.dones[step] = self.next_done
            
            # Get action from policy
            if self.args.random_policy:
                step_action = torch.rand((self.args.num_envs, self.envs.num_actions), device=self.device)
            else:
                with torch.no_grad():
                    rollout_obs = self.next_obs if not self.args.stu_train else self.tea_next_obs
                    rollout_state = self.next_state if not self.args.stu_train else self.tea_next_state
                    
                    if self.args.use_lstm:
                        step_action, _, logprob, _, value, self.next_lstm_state, value_c = self.rollout_agent.get_action_and_value(
                            rollout_obs, rollout_state, self.next_lstm_state, self.next_done
                        )
                    else:
                        step_action, _, logprob, _, value, value_c = self.rollout_agent.get_action_and_value(
                            rollout_obs, rollout_state
                        )
                    
                    self.actions[step] = step_action
                    self.logprobs[step] = logprob
                    self.values[step] = value.flatten()
                    if self.args.use_cost:
                        self.values_c[step] = value_c
            
            # Step environment
            next_obs_dict, reward, done, infos = self.envs.step(step_action)
            self.next_obs = next_obs_dict["obs"].to(self.device)
            self.next_state = next_obs_dict["states"].to(self.device)
            self.next_done = done.to(self.device)
            
            if self.args.stu_train:
                self.tea_next_obs = next_obs_dict["fixed_obs"].to(self.device)
                self.tea_next_state = next_obs_dict["fixed_state"].to(self.device)
            
            # Process rewards
            org_reward = reward.to(self.device).view(-1)
            reward = self.reward_normalizer.normalize(org_reward, self.next_done) if self.args.norm_rew else org_reward
            self.rewards[step] = reward
            
            # Process costs
            if self.args.use_cost:
                org_cost = infos["cost"].to(self.device)
                cost = self.cost_normalizer.normalize(org_cost, self.next_done) if self.args.norm_cost else org_cost
                self.costs[step] = cost
            
            # Update episode statistics
            self._update_episode_stats(org_reward, org_cost if self.args.use_cost else 0, infos)
        
        return initial_lstm_state if self.args.use_lstm else None
    
    
    def _update_episode_stats(self, org_reward, org_cost, infos):
        """Update episode statistics when episodes complete."""
        terminal_index = self.next_done == 1
        terminal_nums = terminal_index.sum().item()
        
        self.step_r_store["eps_r"] += org_reward
        self.step_r_store["eps_c"] += org_cost.sum(dim=-1) if self.args.use_cost else 0
        self.step_r_store["eps_scenevel_p"] += infos.get("scene_linvel_penalty", 0)
        self.step_r_store["eps_sceneacc_p"] += infos.get("scene_linacc_penalty", 0)
        self.step_r_store["eps_act_p"] += infos.get("arm_qvel_penalty", 0)
        
        if terminal_nums > 0:
            terminal_ids = terminal_index.nonzero().flatten()
            success_buf = infos["success"][terminal_index]
            success_ids = terminal_ids[success_buf.to(torch.bool)]
            
            self.global_episodes += terminal_nums
            self.curri_episodes += terminal_nums
            self.success_episodes += len(success_ids)
            
            # Update buffers
            for key in self.step_r_store.keys():
                update_tensor_buffer(self.eps_r_store[key], self.step_r_store[key][terminal_index])
                self.step_r_store[key][terminal_index] = 0.
            
            for key in self.eps_r_store.keys():
                if key in self.step_r_store.keys() or key not in infos:
                    continue
                record_index = success_ids if key in self.success_record_keys else terminal_index
                update_tensor_buffer(self.eps_r_store[key], infos[key][record_index])
            
            # Compute averages
            for key in self.eps_r_store.keys():
                valid_episodes = self.success_episodes if key in self.success_record_keys else self.curri_episodes
                self.eps_r_avg[key] = torch.mean(self.eps_r_store[key][-valid_episodes:]).item()
            
            self.cur_rew = self.eps_r_avg["eps_r"] - self.args.successRewardScale * self.eps_r_avg["eps_c"]
            self.cur_success_rate = self.eps_r_avg["success"]
            self.cur_eps_time = self.eps_r_avg["eps_time"]
            self.ready_to_record = self.curri_episodes > self.args.running_len
            
            self.training_info['last_episode'] = {
                'global_iter': self.global_update_iter,
                'global_episodes': self.global_episodes,
                'global_steps': self.global_step,
                'success_rate': self.success_episodes / self.curri_episodes if self.curri_episodes > 0 else 0,
                'reward': self.eps_r_avg['eps_r'],
                'cost': self.eps_r_avg['eps_c'],
                'eps_time': self.eps_r_avg['eps_time'],
                'eps_horizon': self.eps_r_avg['eps_horizon'],
                'eps_max_scevel': self.eps_r_avg['eps_max_scevel'],
            }
    
    
    def compute_advantages(self, initial_lstm_state=None):
        """Compute GAE advantages and returns."""
        with torch.no_grad():
            if self.args.use_lstm:
                next_value, next_value_c = self.agent.get_value(self.next_state, self.next_lstm_state, self.next_done)
            else:
                next_value, next_value_c = self.agent.get_value(self.next_state)
            
            next_value = next_value.flatten()
            
            advantages = torch.zeros_like(self.rewards, device=self.device)
            lastgaelam = 0
            
            if self.args.use_cost:
                advantages_c = torch.zeros_like(self.costs, device=self.device)
                lastgaelam_c = 0
            
            for t in reversed(range(self.args.num_steps)):
                if t == self.args.num_steps - 1:
                    nextnonterminal = 1.0 - self.next_done
                    nextvalues = next_value
                    if self.args.use_cost:
                        nextvalues_c = next_value_c
                else:
                    nextnonterminal = 1.0 - self.dones[t + 1]
                    nextvalues = self.values[t + 1]
                    if self.args.use_cost:
                        nextvalues_c = self.values_c[t + 1]
                
                delta = self.rewards[t] + self.args.gamma * nextvalues * nextnonterminal - self.values[t]
                advantages[t] = lastgaelam = delta + self.args.gamma * self.args.gae_lambda * nextnonterminal * lastgaelam
                
                if self.args.use_cost:
                    delta_c = self.costs[t] + self.c_gamma * nextvalues_c * nextnonterminal.view(-1, 1) - self.values_c[t]
                    advantages_c[t] = lastgaelam_c = delta_c + self.c_gamma * self.args.gae_lambda * nextnonterminal.view(-1, 1) * lastgaelam_c
            
            returns = advantages + self.values
            if self.args.use_cost:
                returns_c = advantages_c + self.values_c
                return returns, advantages, returns_c, advantages_c
            return returns, advantages, None, None
    
    
    def update_policy(self, returns, advantages, returns_c=None, advantages_c=None, initial_lstm_state=None):
        """Update policy using PPO or P3O."""
        # Flatten batches
        obs_shape = self.envs.obs_space.shape
        state_shape = self.envs.state_space.shape
        act_shape = self.envs.act_space.shape if not self.args.meta_rl else (2,)
        
        b_obs = self.obs.reshape((-1,) + obs_shape)
        b_states = self.states.reshape((-1,) + state_shape)
        b_logprobs = self.logprobs.reshape(-1)
        b_actions = self.actions.reshape((-1,) + act_shape)
        b_dones = self.dones.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = self.values.reshape(-1)
        
        if self.args.use_cost:
            b_advantages_c = advantages_c.reshape(-1, self.args.num_cost)
            b_returns_c = returns_c.reshape(-1, self.args.num_cost)
            b_values_c = self.values_c.reshape(-1, self.args.num_cost)
        
        # Normalize advantages
        b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)
        if self.args.use_cost:
            b_adv_c_mean = b_advantages_c.mean(dim=0)
            b_adv_c_std = b_advantages_c.std(dim=0)
            b_advantages_c = (b_advantages_c - b_adv_c_mean) / (b_adv_c_std + 1e-8)
        
        # Prepare indices
        if self.args.use_lstm:
            envsperbatch = self.args.num_envs // self.args.num_minibatches
            envinds = np.arange(self.args.num_envs)
            flatinds = np.arange(self.args.batch_size).reshape(self.args.num_steps, self.args.num_envs)
            end_idx = self.args.num_envs
            step_num = envsperbatch
        else:
            b_inds = np.arange(self.args.batch_size)
            end_idx = self.args.batch_size
            step_num = self.args.minibatch_size
        
        # Save previous parameters for KL divergence check
        if self.args.target_kl is not None:
            agent_params_store = deepcopy(self.agent.state_dict())
            optim_params_store = deepcopy(self.optimizer.state_dict())
        
        policy_diverged = False
        
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
                    _, mu, newlogprob, entropy, newvalue, _, newvalue_c = self.agent.get_action_and_value(
                        b_obs[mb_inds],
                        b_states[mb_inds],
                        (initial_lstm_state[0][:, mbenvinds], initial_lstm_state[1][:, mbenvinds],
                         initial_lstm_state[2][:, mbenvinds], initial_lstm_state[3][:, mbenvinds]),
                        b_dones[mb_inds],
                        b_actions[mb_inds],
                    )
                else:
                    mb_inds = b_inds[start:end]
                    _, mu, newlogprob, entropy, newvalue, newvalue_c = self.agent.get_action_and_value(
                        b_obs[mb_inds],
                        b_states[mb_inds],
                        b_actions[mb_inds]
                    )
                
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()
                mb_advantages = b_advantages[mb_inds]
                
                with torch.no_grad():
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    policy_diverged = self.args.target_kl is not None and approx_kl > self.args.target_kl
                
                # Policy loss
                clipped_ratio = torch.clamp(ratio, 1 - self.args.clip_coef, 1 + self.args.clip_coef)
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * clipped_ratio
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()
                
                # Value loss
                newvalue = newvalue.view(-1)
                if self.args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -self.args.clip_coef,
                        self.args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()
                
                # Cost loss
                if self.args.use_cost:
                    mb_advantages_c = b_advantages_c[mb_inds]
                    cost_loss1 = mb_advantages_c * ratio.view(-1, 1)
                    cost_loss2 = mb_advantages_c * clipped_ratio.view(-1, 1)
                    L_clip_c = torch.max(cost_loss1, cost_loss2).mean(dim=0)
                    
                    batch_cost_ret = (1.0 - self.c_gamma) * b_returns_c[mb_inds].mean(dim=0)
                    batch_cost_ret = (batch_cost_ret + b_adv_c_mean) / (b_adv_c_std + 1e-8)
                    L_viol = L_clip_c + batch_cost_ret
                    L_viol = (self.c_scale * torch.clamp(L_viol, min=0.0)).sum()
                    pg_loss += L_viol
                    
                    # Cost value loss
                    newvalue_c = newvalue_c.view(-1, self.args.num_cost)
                    if self.args.clip_vloss:
                        v_loss_unclipped_c = (newvalue_c - b_returns_c[mb_inds]) ** 2
                        v_clipped_c = b_values_c[mb_inds] + torch.clamp(
                            newvalue_c - b_values_c[mb_inds],
                            -self.args.clip_coef,
                            self.args.clip_coef,
                        )
                        v_loss_clipped_c = (v_clipped_c - b_returns_c[mb_inds]) ** 2
                        v_loss_max_c = torch.max(v_loss_unclipped_c, v_loss_clipped_c)
                        v_loss_c = 0.5 * v_loss_max_c.mean(dim=0).sum()
                    else:
                        v_loss_c = 0.5 * ((newvalue_c - b_returns_c[mb_inds]) ** 2).mean(dim=0).sum()
                    v_loss += v_loss_c
                
                entropy_loss = entropy.mean()
                pg_coef = 0. if self.global_update_iter <= self.args.warmup_iters else 1.
                ent_coef = 0. if self.global_update_iter <= self.args.warmup_iters else 1.
                loss = pg_coef * pg_loss + self.args.vf_coef * v_loss - ent_coef * self.cur_ent * entropy_loss
                
                if not self.args.beta:
                    loss += self.args.bounds_loss_coef * bound_loss(mu, soft_bound=1.)
                
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.agent.critic.parameters(), self.args.max_grad_norm)
                nn.utils.clip_grad_norm_(self.agent.actor.parameters(), self.args.max_grad_norm)
                if self.args.use_cost:
                    nn.utils.clip_grad_norm_(self.agent.critic_inst.parameters(), self.args.max_grad_norm)
                    nn.utils.clip_grad_norm_(self.agent.critic_t.parameters(), self.args.max_grad_norm)
                self.optimizer.step()
                
                if policy_diverged:
                    break
                
                if self.args.anneal_lr:
                    if self.args.scheduler == 'adapt':
                        new_lr = self.lr_scheduler.update(self.optimizer.param_groups[0]["lr"], approx_kl)
                    else:
                        new_lr, _ = self.lr_scheduler.update(self.global_step)
                    self.optimizer.param_groups[0]["lr"] = new_lr
            
            if policy_diverged:
                break
        
        if policy_diverged:
            self.agent.load_state_dict(agent_params_store)
            self.optimizer.load_state_dict(optim_params_store)
            self.skipped_update_iter += 1
            
            if self.args.saving and self.args.wandb:
                wandb.log({
                    'debug/skipped_update_iter': self.skipped_update_iter,
                    'debug/skipped_kl': approx_kl.item(),
                    'debug/skipped_adv': mb_advantages.mean().item(),
                    'debug/skipped_ratio': ratio.mean().item(),
                    'debug/skipped_entropy': entropy_loss.item(),
                })
            return True
        
        # Compute explained variance
        y_pred = b_values.to(torch.float32).cpu().numpy()
        y_true = b_returns.to(torch.float32).cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
        
        # Log metrics
        self._log_training_metrics(pg_loss, v_loss, entropy_loss, approx_kl, mb_advantages, explained_var, 
                                   v_loss_c if self.args.use_cost else None,
                                   L_viol if self.args.use_cost else None)
        
        return False
    
    
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
    
    
    def update_curriculum(self):
        """Update curriculum learning parameters (for vanilla policy training)."""
        self.cur_ent = linearAmplifier(*self.args.ent_coef, self.global_step, self.args.total_timesteps, self.args.curr_rate)
        self.envs.cfg['r_epstime_scale'] = linearAmplifier(*self.args.epstimeRewardScale, self.global_step, self.args.total_timesteps, self.args.curr_rate)
        self.envs.cfg['r_scene_vel_scale'] = linearAmplifier(*self.args.scevelRewardScale, self.global_step, self.args.total_timesteps, self.args.curr_rate)
        
        if self.args.pre_train and self.ready_to_record and self.args.success_threshold > 0:
            if self.eps_r_avg['success'] >= self.args.success_threshold:
                self.curriculum_above += 1
                self.curriculum_below = 0
                if self.curriculum_above >= self.args.curri_hold_iters:
                    self.curri_ratio = min(self.curri_ratio + self.args.curriculum_step, 1.0)
                    self.envs.update_dr_params(self.curri_ratio)
                    self.curriculum_above = 0
            else:
                self.curriculum_below += 1
                self.curriculum_above = 0
                if self.curriculum_below >= self.args.curri_hold_iters:
                    self.curri_ratio = max(self.curri_ratio - self.args.curriculum_step, 0.0)
                    self.envs.update_dr_params(self.curri_ratio)
                    self.curriculum_below = 0
    
    
    def log_episode_metrics(self):
        """Log episode-level metrics to wandb."""
        if not (self.args.saving and not self.args.stu_train):
            return
        
        if self.args.wandb:
            wandb.log({
                'misc/global_episodes': self.global_episodes,
                'misc/global_steps': self.global_step,
                'misc/global_iterations': self.global_update_iter
            }, commit=False)
            
            wandb_logging = {}
            for key in self.eps_r_avg.keys():
                if key == "success":
                    continue
                wandb_logging[f"reward/{key}"] = self.eps_r_avg[key]
            wandb_logging['reward/curriculum_ratio'] = self.curri_ratio
            
            if self.ready_to_record:
                if self.avg_buffer_reset:
                    self.reward_episodes += self.curri_episodes
                    self.reward_update_iters += self.curri_update_iters
                    self.reward_steps += self.curri_steps
                    self.avg_buffer_reset = False
                wandb_logging.update({
                    'misc/s_episodes': self.global_episodes - self.reward_episodes,
                    'misc/s_iterations': self.global_update_iter - self.reward_update_iters,
                    'misc/s_steps': self.global_step - self.reward_steps,
                    'reward/success_rate': self.eps_r_avg["success"]
                })
            wandb.log(wandb_logging)
    
    
    def _log_training_metrics(self, pg_loss, v_loss, entropy_loss, approx_kl, mb_advantages, explained_var, v_loss_c=None, L_viol=None):
        """Log training metrics to wandb."""
        if not (self.args.saving and self.args.wandb):
            return
        
        if self.args.beta:
            concentration_alpha = self.agent.probs.concentration0.mean(dim=0)
            concentration_beta = self.agent.probs.concentration1.mean(dim=0)
            entropy_log = self.agent.prob_entropy.mean(dim=0)
            
            wandb.log({
                'entropy/entropy': entropy_loss.item(),
                'entropy/entropy_x': entropy_log[0].item(),
                'entropy/entropy_y': entropy_log[1].item(),
                'entropy/entropy_z': entropy_log[2].item(),
                'entropy/entropy_Rz': entropy_log[3].item(),
                'concentration_a/alpha_x': concentration_alpha[0].item(),
                'concentration_a/alpha_y': concentration_alpha[1].item(),
                'concentration_a/alpha_z': concentration_alpha[2].item(),
                'concentration_a/alpha_Rz': concentration_alpha[3].item(),
                'concentration_b/beta_x': concentration_beta[0].item(),
                'concentration_b/beta_y': concentration_beta[1].item(),
                'concentration_b/beta_z': concentration_beta[2].item(),
                'concentration_b/beta_Rz': concentration_beta[3].item(),
            }, commit=False)
        else:
            entropy_log = self.agent.prob_entropy.mean(dim=0)
            act_mu_log = self.agent.probs.mean
            wandb.log({
                'entropy/entropy': entropy_loss.item(),
                'entropy/entropy_x': entropy_log[0].item(),
                'entropy/entropy_y': entropy_log[1].item(),
                'entropy/entropy_z': entropy_log[2].item(),
                'entropy/entropy_Rz': entropy_log[3].item(),
                'action/max_mu_x': act_mu_log.max().item(),
                'action/min_mu_x': act_mu_log.min().item(),
            }, commit=False)
        
        if self.args.use_cost:
            wandb.log({
                'train/critic_cost_loss': v_loss_c.item(),
                'train/actor_cost_loss': L_viol.item(),
            }, commit=False)
        
        wandb.log({
            'steps': self.global_step,
            'iterations': self.global_update_iter,
            'train/learning_rate': self.optimizer.param_groups[0]["lr"],
            'train/critic_loss': v_loss.item(),
            'train/policy_loss': pg_loss.item(),
            'train/approx_kl': approx_kl.item(),
            'train/advantages': mb_advantages.mean().item(),
            'train/explained_variance': explained_var,
            'train/entropy_coef': self.cur_ent,
            'train/epstimeRewardScale': self.envs.cfg['r_epstime_scale'],
            'train/scevelRewardScale': self.envs.cfg['r_scene_vel_scale']
        })
    
    
    def save_checkpoints(self):
        """Save model checkpoints based on performance."""
        if not (self.args.saving and not self.args.stu_train):
            return
        
        if self.ready_to_record and self.curri_ratio == 1:
            if self.cur_rew >= self.best_rew:
                self.best_rew = self.cur_rew
                best_rew_iter = self.global_update_iter
                self.training_info['best_rew'] = {
                    'iteration': best_rew_iter,
                    'reward': self.best_rew,
                    'success_rate': self.cur_success_rate,
                    'pure_reward': self.eps_r_avg['eps_r'],
                    'cost': self.eps_r_avg['eps_c']
                }
                self.agent.save_checkpoint(folder_path=self.args.checkpoint_dir, suffix='best_rew', reward_normalizer=self.reward_normalizer)
            
            if self.cur_success_rate >= self.best_success_rate:
                self.best_success_rate = self.cur_success_rate
                best_suc_iter = self.global_update_iter
                self.training_info['best_suc'] = {
                    'iteration': best_suc_iter,
                    'reward': self.cur_rew,
                    'success_rate': self.best_success_rate,
                    'pure_reward': self.eps_r_avg['eps_r'],
                    'cost': self.eps_r_avg['eps_c']
                }
                self.agent.save_checkpoint(folder_path=self.args.checkpoint_dir, suffix='best_suc', reward_normalizer=self.reward_normalizer)
            
            cur_local_success = self.success_episodes / self.curri_episodes if self.curri_episodes > 0 else 0
            if (self.cur_success_rate >= self.args.init_success and 
                cur_local_success >= self.args.init_success and 
                self.cur_eps_time >= self.max_eps_time and 
                self.args.pre_train):
                self.max_eps_time = self.cur_eps_time
                max_eps_time_iter = self.global_update_iter
                self.training_info['max_eps_time'] = {
                    'iteration': max_eps_time_iter,
                    'eps_time': self.max_eps_time
                }
                self.agent.save_checkpoint(folder_path=self.args.checkpoint_dir, suffix='init', reward_normalizer=self.reward_normalizer)
        
        if self.global_update_iter % self.args.record_iter == 0 and self.global_update_iter > 0:
            self.training_info['last_ckpt_iter'] = self.global_update_iter
            if self.args.last_only:
                self.agent.save_checkpoint(folder_path=self.args.checkpoint_dir, suffix='last', reward_normalizer=self.reward_normalizer)
            elif not self.args.best_only:
                self.agent.save_checkpoint(folder_path=self.args.checkpoint_dir, suffix=str(self.global_update_iter), reward_normalizer=self.reward_normalizer)
        
        save_json(self.meta_data, os.path.join(self.args.trajectory_dir, "meta_data.json"))
    
    
    def print_status(self, update, num_updates):
        """Print training status."""
        print_msg = (f"Current Iteration: {update}/{num_updates} | Episodes: {self.global_episodes} | "
                    f"Reward: {self.cur_rew:.3f}/{self.best_rew:.3f} | "
                    f"Success Rate: {self.cur_success_rate:.4f}/{self.best_success_rate:.4f}")
        
        if self.args.pre_train:
            print_msg += f" | Max Episode Time: {self.cur_eps_time:.3f}/{self.max_eps_time:.3f}"
        if self.args.use_cost:
            print_msg += f" | Cost: {self.eps_r_avg['eps_c']:.3f}"
        
        if not self.args.stu_train:
            print(print_msg + '\r', end='')
    
    
    def print_student_status(self, update, num_updates, ratio_loss, v_loss):
        """Print student training status."""
        print_msg = (f"Current Iteration: {update}/{num_updates} | Episodes: {self.global_episodes} | "
                    f"Reward: {self.cur_rew:.3f}/{self.best_rew:.3f} | "
                    f"Success Rate: {self.cur_success_rate:.4f}/{self.best_success_rate:.4f}")
        
        if self.args.use_cost:
            print_msg += f" | Cost: {self.eps_r_avg['eps_c']:.3f}"
        
        print_msg += f" | BCLoss: {self.cur_loss:.3f}/{self.best_loss:.3f} | Ratio Loss: {ratio_loss:.3f} | Value Loss: {v_loss:.3f}"
        print(print_msg + '\r', end='')
    
    
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


if __name__ == "__main__":
    args = parse_args()
    
    if args.saving:
        # Save the training configuration
        with open(args.json_file_path, 'w') as json_obj:
            json.dump(vars(args), json_obj, indent=4)
    
    trainer = PPOTrainer(args)
    trainer.train()