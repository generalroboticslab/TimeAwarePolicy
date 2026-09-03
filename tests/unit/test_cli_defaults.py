import sys

from projects.TimeAwarePolicy.arguments.training import parse_args


def parse_training_args(monkeypatch, *arguments, task_name="FrankaCubeStack"):
    monkeypatch.setattr(
        sys,
        "argv",
        ["train.py", "--task_name", task_name, *arguments],
    )
    return parse_args()


def write_parent_checkpoint(tmp_path, task_name="FrankaCubeStack"):
    checkpoint = (
        tmp_path / task_name / "parent" / "config.json"
    )
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_text(
        '{\"checkpoint\": null, \"index_episode\": \"best\", '
        '\"control_type\": \"ik\"}'
    )
    return checkpoint


def test_student_training_enables_time_ratio_by_default(monkeypatch, tmp_path):
    write_parent_checkpoint(tmp_path)
    args = parse_training_args(
        monkeypatch,
        "--stu_train",
        "--checkpoint",
        "parent",
        "--train_res_dir",
        str(tmp_path),
    )
    assert args.time_ratio is True


def test_nonstudent_training_keeps_time_ratio_disabled(monkeypatch):
    args = parse_training_args(monkeypatch)
    assert args.time_ratio is False
    assert args.cube_gripper_clearance_m == 0.05
    assert not hasattr(args, "cabinet_contact_force_limit_n")


def test_public_ppo_defaults_match_manuscript_table(monkeypatch):
    args = parse_training_args(monkeypatch)

    assert args.num_updates == 2500
    assert args.total_timesteps == 2500 * 16384 * 32
    assert args.lr == 2e-4
    assert args.num_envs == 16384
    assert args.control_freq_inv == 3
    assert 1 / args.dt / args.control_freq_inv == 20
    assert args.num_steps == 32
    assert args.num_envs * args.num_steps == 524288
    assert args.minibatch_size == 131072
    assert args.update_epochs == 5
    assert args.gamma == 0.995
    assert args.gae_lambda == 0.95
    assert args.clip_coef == 0.2
    assert args.vf_coef == 0.5
    assert args.ent_coef == [0.005, 0.005]
    assert args.max_grad_norm == 0.5
    assert args.target_kl == 2.5
    assert args.hidden_size == [256, 128, 64]
    assert args.successRewardScale == 1000


def test_default_update_budgets_are_task_and_stage_specific(monkeypatch, tmp_path):
    initial_budgets = {
        "FrankaCubeStack": 2500,
        "FrankaGmPour": 6000,
        "FrankaCabinet": 2500,
    }
    checkpoint_budgets = {
        "FrankaCubeStack": 1500,
        "FrankaGmPour": 2500,
        "FrankaCabinet": 1500,
    }

    for task_name, expected_updates in initial_budgets.items():
        args = parse_training_args(monkeypatch, task_name=task_name)
        assert args.num_updates == expected_updates
        assert args.total_timesteps == expected_updates * args.num_envs * args.num_steps

    for task_name, expected_updates in checkpoint_budgets.items():
        write_parent_checkpoint(tmp_path, task_name)
        args = parse_training_args(
            monkeypatch,
            "--checkpoint",
            "parent",
            "--train_res_dir",
            str(tmp_path),
            task_name=task_name,
        )
        assert args.num_updates == expected_updates
        assert args.total_timesteps == expected_updates * args.num_envs * args.num_steps


def test_time_aware_horizons_are_task_specific(monkeypatch, tmp_path):
    horizons = {
        "FrankaCubeStack": 2000,
        "FrankaGmPour": 1600,
        "FrankaCabinet": 2600,
    }

    for task_name, expected_horizon in horizons.items():
        write_parent_checkpoint(tmp_path, task_name)
        args = parse_training_args(
            monkeypatch,
            "--checkpoint",
            "parent",
            "--train_res_dir",
            str(tmp_path),
            "--fixed_configs",
            "--ratio_range",
            "[0.2, 1]",
            task_name=task_name,
        )
        assert args.episodeLength == expected_horizon


def test_explicit_transition_budget_bypasses_default_updates(monkeypatch):
    args = parse_training_args(
        monkeypatch,
        "--total_timesteps",
        "123456789",
    )

    assert args.num_updates is None
    assert args.total_timesteps == 123456789


def test_explicit_updates_override_transition_budget(monkeypatch):
    args = parse_training_args(
        monkeypatch,
        "--num_updates",
        "7",
        "--total_timesteps",
        "123456789",
    )

    assert args.num_updates == 7
    assert args.total_timesteps == 7 * args.num_envs * args.num_steps


def test_public_cmdp_settings_match_manuscript_overrides(monkeypatch):
    args = parse_training_args(
        monkeypatch,
        "--use_cost",
        "--gamma",
        "1.0",
        "--epstimeRewardScale",
        "[100, 100]",
    )

    assert args.gamma == 1.0
    assert args.c_gamma == [1, 0.99]
    assert args.c_scale == [0, 1]
    assert args.epstimeRewardScale == [100, 100]
    assert args.vf_coef == 0.5
    assert args.lagrangian_init == 1.0
    assert args.lagrangian_lr == 0.05
    assert args.lagrangian_max == 100.0
    assert args.cpo_max_kl == 0.01
    assert args.cpo_cg_iters == 10
    assert args.cpo_cg_damping == 0.1
    assert args.cpo_backtrack_iters == 10
    assert args.cpo_backtrack_coeff == 0.8


def test_cube_threshold_override_is_explicit(monkeypatch):
    args = parse_training_args(
        monkeypatch,
        "--cube_gripper_clearance_m",
        "0.05",
    )
    assert args.cube_gripper_clearance_m == 0.05


def test_explicit_time_ratio_override_is_preserved(monkeypatch, tmp_path):
    write_parent_checkpoint(tmp_path)
    args = parse_training_args(
        monkeypatch,
        "--stu_train",
        "--time_ratio",
        "false",
        "--checkpoint",
        "parent",
        "--train_res_dir",
        str(tmp_path),
    )
    assert args.time_ratio is False
