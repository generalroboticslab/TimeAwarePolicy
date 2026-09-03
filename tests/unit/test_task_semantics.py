from pathlib import Path

from envs.isaacgymenvs.task_semantics import (
    DEFAULT_CUBE_GRIPPER_CLEARANCE_M,
    apply_no_dense_reward_semantics,
    cube_gripper_clearance,
)


ROOT = Path(__file__).resolve().parents[2]


def test_no_dense_removes_all_task_shaping_including_gmpour_hold_reward():
    reward_settings = {
        "r_dist_scale": 0.025,
        "r_lift_scale": 0.5,
        "r_align_scale": 2.0,
        "r_hold_scale": 5.0,
        "r_success": 1000.0,
        "r_epstime_scale": 100.0,
        "r_force_penalty_scale": 0.01,
    }
    task_dense_reward_keys = (
        "r_dist_scale",
        "r_lift_scale",
        "r_align_scale",
    )

    initial_policy_settings = dict(reward_settings)
    apply_no_dense_reward_semantics(
        initial_policy_settings,
        task_dense_reward_keys,
        no_dense=False,
    )
    assert initial_policy_settings["r_hold_scale"] == 5.0

    sparse_settings = dict(reward_settings)
    apply_no_dense_reward_semantics(
        sparse_settings,
        task_dense_reward_keys,
        no_dense=True,
    )
    for key in (*task_dense_reward_keys, "r_hold_scale"):
        assert sparse_settings[key] == 0.0
    assert sparse_settings["r_success"] == 1000.0
    assert sparse_settings["r_epstime_scale"] == 100.0
    assert sparse_settings["r_force_penalty_scale"] == 0.01


def test_cube_success_uses_manuscript_five_centimeter_default():
    source = (ROOT / "envs/isaacgymenvs/tasks/franka_cube_stack.py").read_text()
    assert DEFAULT_CUBE_GRIPPER_CLEARANCE_M == 0.05
    assert cube_gripper_clearance({}) == 0.05
    assert cube_gripper_clearance({"cube_gripper_clearance_m": 0.05}) == 0.05
    assert cube_gripper_clearance({"away_dist": 0.07}) == 0.07
    assert "cube_gripper_clearance(self.cfg)" in source


def test_common_noise_and_gripper_delay_match_manuscript():
    source = (
        ROOT / "envs/isaacgymenvs/tasks/base/vec_task.py"
    ).read_text()

    assert '"eef_pos": [0., 0.01]' in source
    assert '"gripper_delay": [0., 0.1]' in source
    assert '"gripper_delay": [0.2, 0.2]' in source
    assert "torch_rand_float(-1, 1" in source


def test_cabinet_has_no_contact_force_failure_path():
    source = (ROOT / "envs/isaacgymenvs/tasks/franka_cabinet.py").read_text()
    assert "cabinet_contact_force_limit_n" not in source
    assert "cabinet_contact_forces" not in source


def test_gmpour_loads_packaged_cup_assets_without_rewriting_them():
    source = (ROOT / "envs/isaacgymenvs/tasks/franka_gm_pour.py").read_text()
    cup_b = (ROOT / "envs/assets/urdf/procedural/cupB.urdf").read_text()

    assert '"urdf/procedural/cup.urdf"' in source
    assert '"urdf/procedural/cupB.urdf"' in source
    assert "create_hollow_cylinder(" not in source
    assert "create_hollow_cylinder_mesh(" not in source
    assert '<box size="0.1 0.1 0.005" />' in cup_b
    assert 'xyz="0.05 0 0"' in cup_b


def test_cube_and_gmpour_actor_observations_do_not_add_target_orientation():
    cases = {
        "franka_cube_stack.py": (
            'obs_names = ["cubeA_pos", "cubeA_quat", '
            '"cubeA_to_cubeB_pos", "eef_pos", "eef_quat"]'
        ),
        "franka_gm_pour.py": (
            'obs_names = ["cupA_rimpos", "cupA_rimquat", '
            '"cupA_to_cupB_pos", "eef_pos", "eef_quat"]'
        ),
    }
    task_root = ROOT / "envs/isaacgymenvs/tasks"

    for file_name, expected_actor_observations in cases.items():
        source = (task_root / file_name).read_text()
        assert expected_actor_observations in source
        assert 'self.cfg["env"]["numObservations"] = 18' in source

    cube_source = (task_root / "franka_cube_stack.py").read_text()
    gm_source = (task_root / "franka_gm_pour.py").read_text()
    assert 'obs_names = ["cubeB_quat"' not in cube_source
    assert 'obs_names = ["cupB_quat"' not in gm_source
