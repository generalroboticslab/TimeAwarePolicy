"""Checkpoint and initializer-quality snapshot management."""

import os

from core.common.io import save_json


class CheckpointingMixin:
    def save_checkpoints(self):
        """Save model checkpoints based on performance."""
        if not (self.args.saving and not self.args.stu_train):
            return

        if self.ready_to_record and self.curri_ratio == 1:
            if self.cur_checkpoint_score >= self.best_checkpoint_score:
                self.best_checkpoint_score = self.cur_checkpoint_score
                best_rew_iter = self.attempted_update_iter
                self.training_info["best_rew"] = {
                    "iteration": best_rew_iter,
                    "accepted_iteration": self.global_update_iter,
                    "checkpoint_score": self.best_checkpoint_score,
                    "reward": self.best_checkpoint_score,
                    "success_rate": self.cur_success_rate,
                    "pure_reward": self.eps_r_avg["eps_r"],
                    "cost": self.eps_r_avg["eps_c"],
                }
                self.agent.save_checkpoint(
                    folder_path=self.args.checkpoint_dir,
                    suffix="best_rew",
                    reward_normalizer=self.reward_normalizer,
                )

            if self.cur_success_rate >= self.best_success_rate:
                self.best_success_rate = self.cur_success_rate
                best_suc_iter = self.attempted_update_iter
                self.training_info["best_suc"] = {
                    "iteration": best_suc_iter,
                    "accepted_iteration": self.global_update_iter,
                    "checkpoint_score": self.cur_checkpoint_score,
                    "reward": self.cur_checkpoint_score,
                    "success_rate": self.best_success_rate,
                    "pure_reward": self.eps_r_avg["eps_r"],
                    "cost": self.eps_r_avg["eps_c"],
                }
                self.agent.save_checkpoint(
                    folder_path=self.args.checkpoint_dir,
                    suffix="best_suc",
                    reward_normalizer=self.reward_normalizer,
                )

            cur_local_success = (
                self.success_episodes / self.curri_episodes
                if self.curri_episodes > 0
                else 0
            )
            if (
                self.cur_success_rate >= self.args.init_success
                and cur_local_success >= self.args.init_success
                and self.cur_eps_time >= self.max_eps_time
                and self.args.pre_train
            ):
                self.max_eps_time = self.cur_eps_time
                self.training_info["max_eps_time"] = {
                    "iteration": self.global_update_iter,
                    "eps_time": self.max_eps_time,
                }
                self.agent.save_checkpoint(
                    folder_path=self.args.checkpoint_dir,
                    suffix="init",
                    reward_normalizer=self.reward_normalizer,
                )

        if self.global_update_iter % self.args.record_iter == 0 and self.global_update_iter > 0:
            self.training_info["last_ckpt_iter"] = self.global_update_iter
            if self.args.last_only:
                self.agent.save_checkpoint(
                    folder_path=self.args.checkpoint_dir,
                    suffix="last",
                    reward_normalizer=self.reward_normalizer,
                )
            elif not self.args.best_only:
                self.agent.save_checkpoint(
                    folder_path=self.args.checkpoint_dir,
                    suffix=str(self.global_update_iter),
                    reward_normalizer=self.reward_normalizer,
                )

        save_json(
            self.meta_data,
            os.path.join(self.args.trajectory_dir, "meta_data.json"),
        )

    def save_quality_candidate(self):
        """Save snapshots for held-out initializer-quality selection."""
        interval = self.args.quality_candidate_interval
        if not (self.args.saving and self.args.pre_train and interval > 0):
            return False

        if self.quality_candidate_start_update is None:
            start_ready = (
                self.ready_to_record
                and self.curri_ratio == 0
                and self.cur_success_rate
                >= self.args.quality_candidate_start_success
            )
            if not start_ready:
                return False
            self.quality_candidate_start_update = self.global_update_iter
            self.meta_data["quality_candidate_protocol"] = {
                "start_accepted_update": self.global_update_iter,
                "start_attempted_update": self.attempted_update_iter,
                "start_rolling_success_rate": self.cur_success_rate,
                "start_curriculum_ratio": self.curri_ratio,
                "interval_accepted_updates": interval,
                "selection_status": "requires_held_out_full_dr_evaluation",
            }

        offset = self.global_update_iter - self.quality_candidate_start_update
        if offset % interval != 0:
            return False
        if self.quality_candidate_last_update == self.global_update_iter:
            return False

        suffix = f"candidate_u{self.global_update_iter:05d}"
        self.agent.save_checkpoint(
            folder_path=self.args.checkpoint_dir,
            suffix=suffix,
            reward_normalizer=self.reward_normalizer,
        )
        self.quality_candidate_last_update = self.global_update_iter
        self.quality_candidates[suffix] = {
            "accepted_update": self.global_update_iter,
            "attempted_update": self.attempted_update_iter,
            "rolling_success_rate": self.cur_success_rate,
            "curriculum_ratio": self.curri_ratio,
            "global_steps": self.global_step,
            "label_status": "unlabeled_pending_strict_full_dr_evaluation",
        }
        save_json(
            self.meta_data,
            os.path.join(self.args.trajectory_dir, "meta_data.json"),
        )
        return True
