"""Curriculum updates for policy training."""

from core.agents.utils import linearAmplifier


class CurriculumMixin:
    def update_curriculum(self):
        """Update curriculum learning parameters for initializer training."""
        self.cur_ent = linearAmplifier(
            *self.args.ent_coef,
            self.global_step,
            self.args.total_timesteps,
            self.args.curr_rate,
        )
        self.envs.cfg["r_epstime_scale"] = linearAmplifier(
            *self.args.epstimeRewardScale,
            self.global_step,
            self.args.total_timesteps,
            self.args.curr_rate,
        )
        self.envs.cfg["r_scene_vel_scale"] = linearAmplifier(
            *self.args.scevelRewardScale,
            self.global_step,
            self.args.total_timesteps,
            self.args.curr_rate,
        )

        if self.args.pre_train and self.ready_to_record and self.args.success_threshold > 0:
            if self.eps_r_avg["success"] >= self.args.success_threshold:
                self.curriculum_above += 1
                self.curriculum_below = 0
                if self.curriculum_above >= self.args.curri_hold_iters:
                    self.curri_ratio = min(
                        self.curri_ratio + self.args.curriculum_step, 1.0
                    )
                    self.envs.update_dr_params(self.curri_ratio)
                    self.curriculum_above = 0
            else:
                self.curriculum_below += 1
                self.curriculum_above = 0
                if self.curriculum_below >= self.args.curri_hold_iters:
                    self.curri_ratio = max(
                        self.curri_ratio - self.args.curriculum_step, 0.0
                    )
                    self.envs.update_dr_params(self.curri_ratio)
                    self.curriculum_below = 0
