from .utils.visualization import BaseSimulatorViewer
from .utils.perlin import perlin_1d, perlin_2d

from .tasks.real.motion_error_measure import MotionErrorMeasure

try:
    from .sims.pyrobot.pyrobot_discrete_action import PyRobotDiscreteActions
except ImportError as e:
    pyrobot_import_error = e
    class PyRobotDiscreteActions:
        def __init__(*args, **kwargs):
            raise pyrobot_import_error

try:
    from .sims.habitat_simulator.realistic_depth_sensor import RealisticHabitatSimDepthSensor
except ImportError as e:
    habitat_sim_import_error = e
    class RealisticHabitatSimDepthSensor:
        def __init__(*args, **kwargs):
            raise habitat_sim_import_error

try:
    from .sims.ros.rosrobot_sim import ROSRobot
    from .sims.ros.intf_node import HabitatInterfaceROSNode
    from .sims.ros.default_cfg import merge_ros_config, get_config
    from .envs.ros_env import ROSEnv, ROSNavRLEnv
except ImportError as e:
    ros_import_error = e
    class ROSRobot:
        def __init__(self, *args, **kwargs):
            raise ros_import_error
    class HabitatInterfaceROSNode:
        def __init__(self, *args, **kwargs):
            raise ros_import_error
    def merge_ros_config(*args, **kwargs):
        raise ros_import_error
    def get_config(*args, **kwargs):
        raise ros_import_error
    class ROSEnv:
        def __init__(self, *args, **kwargs):
            raise ros_import_error
    class ROSNavRLEnv:
        def __init__(self, *args, **kwargs):
            raise ros_import_error

try:
    from .sims.jetbot.jetbot_sim import Jetbot
except ImportError as e:
    jetbot_import_error = e
    class Jetbot:
        def __init__(self, *args, **kwargs):
            raise jetbot_import_error
