"""Rollout collection, episode aggregation, and GAE computation."""

import torch

from core.agents.agent import update_tensor_buffer
from core.agents.utils import successful_episode_indices


class RolloutMixin:
    def collect_rollout(self):
        """Collect rollout data from the vectorized environment."""
        if self.args.use_lstm:
            initial_lstm_state = [
                lstm_state.clone() for lstm_state in self.next_lstm_state
            ]

        for step in range(self.args.num_steps):
            self.global_step += self.args.num_envs
            self.obs[step] = self.next_obs
            self.states[step] = self.next_state
            self.dones[step] = self.next_done
            self.timeouts[step] = self.next_timeout

            if self.args.random_policy:
                step_action = torch.rand(
                    (self.args.num_envs, self.envs.num_actions),
                    device=self.device,
                )
            else:
                with torch.no_grad():
                    rollout_obs = (
                        self.next_obs
                        if not self.args.stu_train
                        else self.tea_next_obs
                    )
                    rollout_state = (
                        self.next_state
                        if not self.args.stu_train
                        else self.tea_next_state
                    )

                    if self.args.use_lstm:
                        (
                            step_action,
                            _,
                            logprob,
                            _,
                            value,
                            self.next_lstm_state,
                            value_c,
                        ) = self.rollout_agent.get_action_and_value(
                            rollout_obs,
                            rollout_state,
                            self.next_lstm_state,
                            self.next_done,
                        )
                    else:
                        step_action, _, logprob, _, value, value_c = (
                            self.rollout_agent.get_action_and_value(
                                rollout_obs, rollout_state
                            )
                        )

                    self.actions[step] = step_action
                    self.logprobs[step] = logprob
                    self.values[step] = value.flatten()
                    if self.args.use_cost:
                        self.values_c[step] = value_c

                    env_action = step_action
                    if self.args.stu_train and self.args.student_onpolicy_rollout:
                        env_action, _, _, _, _, _ = self.agent.get_action_and_value(
                            self.next_obs, self.next_state
                        )

            next_obs_dict, reward, done, infos = self.envs.step(
                env_action if not self.args.random_policy else step_action
            )
            self.next_obs = next_obs_dict["obs"].to(self.device)
            self.next_state = next_obs_dict["states"].to(self.device)
            self.next_done = done.to(self.device)
            if "time_outs" in infos:
                self.next_timeout = infos["time_outs"].to(self.device).float()

            if self.args.stu_train:
                self.tea_next_obs = next_obs_dict["fixed_obs"].to(self.device)
                self.tea_next_state = next_obs_dict["fixed_state"].to(self.device)

            org_reward = reward.to(self.device).view(-1)
            reward = (
                self.reward_normalizer.normalize(org_reward, self.next_done)
                if self.args.norm_rew
                else org_reward
            )
            self.rewards[step] = reward

            if self.args.use_cost:
                org_cost = infos["cost"].to(self.device)
                cost = (
                    self.cost_normalizer.normalize(org_cost, self.next_done)
                    if self.args.norm_cost
                    else org_cost
                )
                self.costs[step] = cost

            self._update_episode_stats(
                org_reward,
                org_cost if self.args.use_cost else 0,
                infos,
            )

        return initial_lstm_state if self.args.use_lstm else None

    def _update_episode_stats(self, org_reward, org_cost, infos):
        """Update rolling episode statistics when episodes complete."""
        terminal_index = self.next_done == 1
        terminal_nums = terminal_index.sum().item()

        self.step_r_store["eps_r"] += org_reward
        self.step_r_store["eps_c"] += (
            org_cost.sum(dim=-1) if self.args.use_cost else 0
        )
        self.step_r_store["eps_scenevel_p"] += infos.get(
            "scene_linvel_penalty", 0
        )
        self.step_r_store["eps_sceneacc_p"] += infos.get(
            "scene_linacc_penalty", 0
        )
        self.step_r_store["eps_act_p"] += infos.get("arm_qvel_penalty", 0)

        if terminal_nums > 0:
            success_ids = successful_episode_indices(
                terminal_index, infos["success"]
            )

            self.global_episodes += terminal_nums
            self.curri_episodes += terminal_nums
            self.success_episodes += len(success_ids)

            for key in self.step_r_store.keys():
                update_tensor_buffer(
                    self.eps_r_store[key],
                    self.step_r_store[key][terminal_index],
                )
                self.step_r_store[key][terminal_index] = 0.0

            for key in self.eps_r_store.keys():
                if key in self.step_r_store.keys() or key not in infos:
                    continue
                record_index = (
                    success_ids
                    if key in self.success_record_keys
                    else terminal_index
                )
                update_tensor_buffer(
                    self.eps_r_store[key], infos[key][record_index]
                )

            for key in self.eps_r_store.keys():
                valid_episodes = (
                    self.success_episodes
                    if key in self.success_record_keys
                    else self.curri_episodes
                )
                self.eps_r_avg[key] = torch.mean(
                    self.eps_r_store[key][-valid_episodes:]
                ).item()

            self.cur_checkpoint_score = self.eps_r_avg["eps_r"]
            self.cur_checkpoint_score -= (
                self.args.c_scale[1]
                * self.args.successRewardScale
                * self.eps_r_avg["eps_c"]
                if self.args.use_cost
                else 0
            )
            self.cur_success_rate = self.eps_r_avg["success"]
            self.cur_eps_time = self.eps_r_avg["eps_time"]
            self.ready_to_record = self.curri_episodes > self.args.running_len

            self.training_info["last_episode"] = {
                "global_iter": self.global_update_iter,
                "attempted_update": self.attempted_update_iter,
                "global_episodes": self.global_episodes,
                "global_steps": self.global_step,
                "success_rate": (
                    self.success_episodes / self.curri_episodes
                    if self.curri_episodes > 0
                    else 0
                ),
                "reward": self.eps_r_avg["eps_r"],
                "cost": self.eps_r_avg["eps_c"],
                "eps_time": self.eps_r_avg["eps_time"],
                "eps_time_p": self.eps_r_avg["eps_time_p"],
                "eps_horizon": self.eps_r_avg["eps_horizon"],
                "eps_max_scevel": self.eps_r_avg["eps_max_scevel"],
            }

    def compute_advantages(self, initial_lstm_state=None):
        """Compute generalized-advantage estimates and returns."""
        with torch.no_grad():
            if self.args.use_lstm:
                next_value, next_value_c = self.agent.get_value(
                    self.next_state,
                    self.next_lstm_state,
                    self.next_done,
                )
            else:
                next_value, next_value_c = self.agent.get_value(self.next_state)

            next_value = next_value.flatten()
            advantages = torch.zeros_like(self.rewards, device=self.device)
            lastgaelam = 0
            if self.args.use_cost:
                advantages_c = torch.zeros_like(self.costs, device=self.device)
                lastgaelam_c = 0

            dones = torch.cat(
                (self.dones, self.next_done.unsqueeze(0)), dim=0
            ) == 1
            timeouts = torch.cat(
                (self.timeouts, self.next_timeout.unsqueeze(0)), dim=0
            ) == 1
            terminates = (
                dones & ~timeouts if self.args.value_bootstrap else dones
            )

            values = torch.cat((self.values, next_value.unsqueeze(0)), dim=0)
            if self.args.use_cost:
                values_c = torch.cat(
                    (self.values_c, next_value_c.unsqueeze(0)), dim=0
                )

            for step in reversed(range(self.args.num_steps)):
                next_nonterminal = 1.0 - terminates[step + 1].float()
                next_values = values[step + 1]
                if self.args.use_cost:
                    next_values_c = values_c[step + 1]

                delta = (
                    self.rewards[step]
                    + self.args.gamma * next_values * next_nonterminal
                    - self.values[step]
                )
                lastgaelam = (
                    delta
                    + self.args.gamma
                    * self.args.gae_lambda
                    * next_nonterminal
                    * lastgaelam
                )
                advantages[step] = lastgaelam = torch.where(
                    dones[step], torch.zeros_like(lastgaelam), lastgaelam
                )

                if self.args.use_cost:
                    nonterminal_column = next_nonterminal.view(-1, 1)
                    delta_c = (
                        self.costs[step]
                        + self.c_gamma * next_values_c * nonterminal_column
                        - self.values_c[step]
                    )
                    lastgaelam_c = (
                        delta_c
                        + self.c_gamma
                        * self.args.gae_lambda
                        * nonterminal_column
                        * lastgaelam_c
                    )
                    advantages_c[step] = lastgaelam_c = torch.where(
                        dones[step].view(-1, 1),
                        torch.zeros_like(lastgaelam_c),
                        lastgaelam_c,
                    )

            returns = advantages + self.values
            if self.args.use_cost:
                returns_c = advantages_c + self.values_c
                return returns, advantages, returns_c, advantages_c
            return returns, advantages, None, None
