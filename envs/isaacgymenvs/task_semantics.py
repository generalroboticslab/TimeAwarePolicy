"""Simulator-independent task success and failure semantics.

Defaults define the public manuscript-aligned task protocol.
"""


DEFAULT_CUBE_GRIPPER_CLEARANCE_M = 0.05


def apply_no_dense_reward_semantics(
    reward_settings,
    task_dense_reward_keys,
    no_dense,
):
    """Disable task shaping rewards while preserving sparse objectives.

    ``no_dense`` removes every task-defined dense reward, including GM Pour's
    shared holding reward. Terminal success/violation rewards, temporal
    objectives, action/force penalties, and CMDP costs remain independent.
    """
    if not no_dense:
        return
    dense_reward_keys = set(task_dense_reward_keys)
    dense_reward_keys.add("r_hold_scale")
    for key in dense_reward_keys:
        if key in reward_settings:
            reward_settings[key] = 0.0


def cube_gripper_clearance(config):
    """Return explicit Cube clearance, defaulting to the manuscript's 5 cm."""
    legacy_override = config.get("away_dist")
    if legacy_override is not None:
        return float(legacy_override)
    return float(config.get(
        "cube_gripper_clearance_m",
        DEFAULT_CUBE_GRIPPER_CLEARANCE_M,
    ))
