from .utils.visualization import BaseSimulatorViewer
from .utils.perlin import perlin_1d, perlin_2d
from .config.default import merge_config, get_config
from .sims.sidechain_sim.sidechain_sim import make_sidechain
from .tasks.nav.noisy_loc_sensors import (NoisyEpisodicCompassSensor,
                                          NoisyEpisodicGPSSensor,
                                          NoisyPointGoalWithGPSAndCompassSensor)
from .tasks.nav.ego_map import EgoMapSensor
from .tasks.real.motion_error_measure import MotionErrorsMeasure
from .tasks.nav.goto_action import GotoAction


try:
    from .sims.pyrobot.pyrobot_discrete_action import PyRobotDiscreteActions
except ImportError as e:
    pyrobot_import_error = e
    class PyRobotDiscreteActions:
        def __init__(*args, **kwargs):
            raise pyrobot_import_error


try:
    from .sims.habitat_simulator.realistic_depth_sensor import RealisticHabitatSimDepthSensor
    from .sims.habitat_simulator.laserscan_sensor import (
        HabitatSimLaserScanSensor, HabitatSimCustom
    )
except ImportError as e:
    habitat_sim_import_error = e
    class RealisticHabitatSimDepthSensor:
        def __init__(*args, **kwargs):
            raise habitat_sim_import_error
    class HabitatSimLaserScanSensor:
        def __init__(*args, **kwargs):
            raise habitat_sim_import_error


try:
    from .sims.ros.rosrobot_sim import ROSRobot
    from .sims.ros.ros_mngr import ROSManager
    from .sims.ros.ros_habsim_sim import ROSHabitatSim
    from .sims.ros.intf_node import HabitatInterfaceROSNode
    from .envs.ros_env import ROSEnv, ROSNavRLEnv
except ImportError as e:
    ros_import_error = e
    class ROSRobot:
        def __init__(self, *args, **kwargs):
            raise ros_import_error
    class ROSManager:
        def __init__(self, *args, **kwargs):
            raise ros_import_error
    class ROSHabitatSim:
        def __init__(self, *args, **kwargs):
            raise ros_import_error
    class HabitatInterfaceROSNode:
        def __init__(self, *args, **kwargs):
            raise ros_import_error
    class ROSEnv:
        def __init__(self, *args, **kwargs):
            raise ros_import_error
    class ROSNavRLEnv:
        def __init__(self, *args, **kwargs):
            raise ros_import_error
    class ROSGotoAction:
        def __init__(self, *args, **kwargs):
            raise ros_import_error


try:
    from .sims.jetbot.jetbot_sim import Jetbot
except ImportError as e:
    jetbot_import_error = e
    class Jetbot:
        def __init__(self, *args, **kwargs):
            raise jetbot_import_error
