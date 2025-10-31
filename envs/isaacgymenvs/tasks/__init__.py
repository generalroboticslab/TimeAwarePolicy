from .franka_cabinet import FrankaCabinet
from .franka_cube_stack import FrankaCubeStack
from .franka_gm_pour import FrankaGmPour


# Mappings from strings to environments
isaacgym_task_map = {
    "FrankaCabinet": FrankaCabinet,
    "FrankaCubeStack": FrankaCubeStack,
    "FrankaGmPour": FrankaGmPour,
}
