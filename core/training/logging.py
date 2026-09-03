"""Training metrics and concise console status output."""

import wandb


class TrainingLoggingMixin:
    def log_episode_metrics(self):
        """Log episode-level metrics to Weights & Biases."""
        if not (self.args.saving and not self.args.stu_train):
            return

        if self.args.wandb:
            wandb.log(
                {
                    "misc/global_episodes": self.global_episodes,
                    "misc/global_steps": self.global_step,
                    "misc/global_iterations": self.global_update_iter,
                    "misc/attempted_updates": self.attempted_update_iter,
                },
                commit=False,
            )

            wandb_logging = {}
            for key in self.eps_r_avg.keys():
                if key == "success":
                    continue
                wandb_logging[f"reward/{key}"] = self.eps_r_avg[key]
            wandb_logging["reward/curriculum_ratio"] = self.curri_ratio
            wandb_logging["reward/rolling_checkpoint_score"] = self.cur_checkpoint_score
            wandb_logging["misc/episode_metric_schema_version"] = 2

            if self.ready_to_record:
                if self.avg_buffer_reset:
                    self.reward_episodes += self.curri_episodes
                    self.reward_update_iters += self.curri_update_iters
                    self.reward_steps += self.curri_steps
                    self.avg_buffer_reset = False
                wandb_logging.update(
                    {
                        "misc/s_episodes": self.global_episodes - self.reward_episodes,
                        "misc/s_iterations": self.global_update_iter - self.reward_update_iters,
                        "misc/s_steps": self.global_step - self.reward_steps,
                        "reward/success_rate": self.eps_r_avg["success"],
                    }
                )
            wandb.log(wandb_logging)

    def _log_training_metrics(
        self,
        pg_loss,
        v_loss,
        entropy_loss,
        approx_kl,
        mb_advantages,
        explained_var,
        v_loss_c=None,
        L_viol=None,
    ):
        """Log optimizer metrics to Weights & Biases."""
        if not (self.args.saving and self.args.wandb):
            return

        if self.args.beta:
            concentration_alpha = self.agent.probs.concentration0.mean(dim=0)
            concentration_beta = self.agent.probs.concentration1.mean(dim=0)
            entropy_log = self.agent.prob_entropy.mean(dim=0)
            wandb.log(
                {
                    "entropy/entropy": entropy_loss.item(),
                    "entropy/entropy_x": entropy_log[0].item(),
                    "entropy/entropy_y": entropy_log[1].item(),
                    "entropy/entropy_z": entropy_log[2].item(),
                    "entropy/entropy_Rz": entropy_log[3].item(),
                    "concentration_a/alpha_x": concentration_alpha[0].item(),
                    "concentration_a/alpha_y": concentration_alpha[1].item(),
                    "concentration_a/alpha_z": concentration_alpha[2].item(),
                    "concentration_a/alpha_Rz": concentration_alpha[3].item(),
                    "concentration_b/beta_x": concentration_beta[0].item(),
                    "concentration_b/beta_y": concentration_beta[1].item(),
                    "concentration_b/beta_z": concentration_beta[2].item(),
                    "concentration_b/beta_Rz": concentration_beta[3].item(),
                },
                commit=False,
            )
        else:
            entropy_log = self.agent.prob_entropy.mean(dim=0)
            act_mu_log = self.agent.probs.mean
            wandb.log(
                {
                    "entropy/entropy": entropy_loss.item(),
                    "entropy/entropy_x": entropy_log[0].item(),
                    "entropy/entropy_y": entropy_log[1].item(),
                    "entropy/entropy_z": entropy_log[2].item(),
                    "entropy/entropy_Rz": entropy_log[3].item(),
                    "action/max_mu_x": act_mu_log.max().item(),
                    "action/min_mu_x": act_mu_log.min().item(),
                },
                commit=False,
            )

        if self.args.use_cost:
            cost_log = {
                "train/critic_cost_loss": v_loss_c.item(),
                "train/actor_cost_loss": L_viol.item(),
            }
            if self.args.cmdp_method == "ppo_lagrangian":
                for index, multiplier in enumerate(self.lagrange_multipliers):
                    cost_log[f"train/lagrange_multiplier_{index}"] = multiplier.item()
            wandb.log(cost_log, commit=False)

        wandb.log(
            {
                "steps": self.global_step,
                "iterations": self.attempted_update_iter,
                "accepted_iterations": self.global_update_iter,
                "train/learning_rate": self.optimizer.param_groups[0]["lr"],
                "train/critic_loss": v_loss.item(),
                "train/policy_loss": pg_loss.item(),
                "train/approx_kl": approx_kl.item(),
                "train/advantages": mb_advantages.mean().item(),
                "train/explained_variance": explained_var,
                "train/entropy_coef": self.cur_ent,
                "train/epstimeRewardScale": self.envs.cfg["r_epstime_scale"],
                "train/scevelRewardScale": self.envs.cfg["r_scene_vel_scale"],
            }
        )

    def print_status(self, update, num_updates):
        """Print training status."""
        print_msg = (
            f"Current Iteration: {self.attempted_update_iter}/{num_updates} | "
            f"Episodes: {self.global_episodes} | "
            f"Checkpoint Score: {self.cur_checkpoint_score:.3f}/{self.best_checkpoint_score:.3f} | "
            f"Success Rate: {self.cur_success_rate:.4f}/{self.best_success_rate:.4f}"
        )
        if self.args.pre_train:
            print_msg += f" | Max Episode Time: {self.cur_eps_time:.3f}/{self.max_eps_time:.3f}"
        if self.args.use_cost:
            print_msg += f" | Cost: {self.eps_r_avg['eps_c']:.3f}"
        if not self.args.stu_train:
            print(print_msg + "\r", end="")

    def print_student_status(self, update, num_updates, ratio_loss, v_loss):
        """Print student-distillation training status."""
        print_msg = (
            f"Current Iteration: {self.attempted_update_iter}/{num_updates} | "
            f"Episodes: {self.global_episodes} | "
            f"Checkpoint Score: {self.cur_checkpoint_score:.3f}/{self.best_checkpoint_score:.3f} | "
            f"Success Rate: {self.cur_success_rate:.4f}/{self.best_success_rate:.4f}"
        )
        if self.args.use_cost:
            print_msg += f" | Cost: {self.eps_r_avg['eps_c']:.3f}"
        print_msg += (
            f" | BCLoss: {self.cur_loss:.3f}/{self.best_loss:.3f} | "
            f"Ratio Loss: {ratio_loss:.3f} | Value Loss: {v_loss:.3f}"
        )
        print(print_msg + "\r", end="")
