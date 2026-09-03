"""Public task, method, and quality labels shared by result scripts."""

from projects.TimeAwarePolicy.paper.style import (
    TimeawareColor,
    TimeInputColor,
    TimeOptimalColor,
    VanillaColor,
)


TASKS = ("cube", "gmpour", "cabinet")
TASK_DISPLAY = {
    "cube": "Cube Stacking",
    "gmpour": "Granular Media Pouring",
    "cabinet": "Drawer Opening",
}
QUALITIES = ("q40", "q60", "q80", "q95")
QUALITY_DISPLAY = {quality: quality.upper() for quality in QUALITIES}
METHODS = ("np3o", "ppo_lagrangian", "cpo")
METHOD_DISPLAY = {
    "np3o": "N-P3O",
    "ppo_lagrangian": "PPO-Lagrangian",
    "cpo": "CPO",
}
QUALITY_COLORS = dict(zip(
    QUALITIES,
    (TimeawareColor, VanillaColor, TimeOptimalColor, TimeInputColor),
))
METHOD_COLORS = dict(zip(
    METHODS,
    (TimeawareColor, VanillaColor, TimeOptimalColor),
))
