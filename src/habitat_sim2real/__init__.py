from .utils.visualization import BaseSimulatorViewer
from .utils.perlin import perlin_1d, perlin_2d

try:
    from .sims.pyrobot.pyrobot_discrete_action import PyRobotDiscreteActions
except ImportError as e:
    class PyRobotDiscreteActions:
        def __init__(*args, **kwargs):
            raise e

try:
    from .sims.habitat_simulator.realistic_depth_sensor import RealisticHabitatSimDepthSensor
except ImportError as e:
    class RealisticHabitatSimDepthSensor:
        def __init__(*args, **kwargs):
            raise e

try:
    from .sims.ros.rosrobot_sim import ROSRobot
    from .sims.ros.intf_node import HabitatInterfaceROSNode
    from .sims.ros.default_cfg import merge_ros_config, get_config
    from .envs.ros_env import ROSEnv, ROSNavRLEnv
except ImportError as e:
    class ROSRobot:
        def __init__(*args, **kwargs):
            raise e
    class HabitatInterfaceROSNode:
        def __init__(*args, **kwargs):
            raise e
    def merge_ros_config(*args, **kwargs):
        raise e
    def get_config(*args, **kwargs):
        raise e
    class ROSEnv:
        def __init__(*args, **kwargs):
            raise e
    class ROSNavRLEnv:
        def __init__(*args, **kwargs):
            raise e

try:
    from .sims.jetbot.hetbot_sim import Jetbot
except ImportError as e:
    class Jetbot:
        def __init__(*args, **kwargs):
            raise e
