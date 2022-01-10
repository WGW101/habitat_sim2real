from typing import Optional, List
import habitat
from habitat.config.default import _C as habitat_defaults


DEFAULT_CFG = habitat.Config()
DEFAULT_CFG.SIMULATOR = habitat.Config()
DEFAULT_CFG.TASK = habitat.Config()

DEFAULT_CFG.SIMULATOR.ROS = habitat.Config()
DEFAULT_CFG.SIMULATOR.ROS.NODE_NAME = "habitat_interface_node"
DEFAULT_CFG.SIMULATOR.ROS.CONNECTION_TIMEOUT = 5.0
DEFAULT_CFG.SIMULATOR.ROS.GETTER_TIMEOUT = 1.0
DEFAULT_CFG.SIMULATOR.ROS.COLOR_IMAGE_TOPIC = "/camera/color/image_raw"
DEFAULT_CFG.SIMULATOR.ROS.DEPTH_IMAGE_TOPIC = "/camera/aligned_depth_to_color/image_raw"
DEFAULT_CFG.SIMULATOR.ROS.IMAGE_SYNC_QUEUE_SIZE = 10
DEFAULT_CFG.SIMULATOR.ROS.MAP_TOPIC = "/map"
DEFAULT_CFG.SIMULATOR.ROS.MAP_FREE_THRESH = 20
DEFAULT_CFG.SIMULATOR.ROS.BUMPER_TOPIC = "/mobile_base/kobuki_node/events/bumper"
DEFAULT_CFG.SIMULATOR.ROS.TF_ROBOT_FRAME = "base_footprint"
DEFAULT_CFG.SIMULATOR.ROS.TF_REF_FRAME = "map"
DEFAULT_CFG.SIMULATOR.ROS.TF_HABITAT_ROBOT_FRAME = "habitat_base_footprint"
DEFAULT_CFG.SIMULATOR.ROS.TF_HABITAT_REF_FRAME = "habitat_map"
DEFAULT_CFG.SIMULATOR.ROS.TF_TIMEOUT = 1.0
DEFAULT_CFG.SIMULATOR.ROS.MOVE_BASE_ACTION_SERVER = "/move_base"
DEFAULT_CFG.SIMULATOR.ROS.MOVE_BASE_PLAN_SERVICE = "/move_base/make_plan"
DEFAULT_CFG.SIMULATOR.ROS.MOVE_BASE_PLAN_TOL = 0.3
DEFAULT_CFG.SIMULATOR.ROS.DYNAMIXEL_SERVICE = "/dynamixel_controller/dynamixel_command"
DEFAULT_CFG.SIMULATOR.ROS.DYNAMIXEL_STATE_TOPIC = "/dynamixel_controller/dynamixel_state"
DEFAULT_CFG.SIMULATOR.ROS.DYNAMIXEL_TICK_PER_RAD = 638
DEFAULT_CFG.SIMULATOR.ROS.DYNAMIXEL_TICK_OFFSET = 2048
DEFAULT_CFG.SIMULATOR.ROS.DYNAMIXEL_TILT_ID = 9
DEFAULT_CFG.SIMULATOR.ROS.DYNAMIXEL_TILT_TOL = 16
DEFAULT_CFG.SIMULATOR.ROS.DYNAMIXEL_TIMEOUT = 5.0
DEFAULT_CFG.SIMULATOR.ROS.SAMPLE_NAV_PT_METHOD = "MAP" # "RVIZ"
DEFAULT_CFG.SIMULATOR.ROS.RVIZ_POINT_TOPIC = "/clicked_point"

DEFAULT_CFG.SIMULATOR.SCAN_SENSOR = habitat.Config()
DEFAULT_CFG.SIMULATOR.SCAN_SENSOR.TYPE = "HabitatSimLaserScanSensor"
DEFAULT_CFG.SIMULATOR.SCAN_SENSOR.MIN_RANGE = 0.0
DEFAULT_CFG.SIMULATOR.SCAN_SENSOR.MAX_RANGE = 100.0
DEFAULT_CFG.SIMULATOR.SCAN_SENSOR.MIN_ANGLE = -3.14159
DEFAULT_CFG.SIMULATOR.SCAN_SENSOR.MAX_ANGLE = 3.14159
DEFAULT_CFG.SIMULATOR.SCAN_SENSOR.INC_ANGLE = 0.0174533
DEFAULT_CFG.SIMULATOR.SCAN_SENSOR.POSITION = [0.0, 0.5, 0.0]
DEFAULT_CFG.SIMULATOR.SCAN_SENSOR.ORIENTATION = [0.0, 0.0, 0.0]
DEFAULT_CFG.SIMULATOR.SCAN_SENSOR.POINTS_FORMAT = "POLAR"
DEFAULT_CFG.SIMULATOR.SCAN_SENSOR.NOISE_RATIO = 0.0

pointgoal_defaults = habitat_defaults.TASK.POINTGOAL_WITH_GPS_COMPASS_SENSOR
DEFAULT_CFG.TASK.NOISY_POINTGOAL_SENSOR = pointgoal_defaults.clone()
DEFAULT_CFG.TASK.NOISY_POINTGOAL_SENSOR.TYPE = "NoisyPointGoalWithGPSAndCompassSensor"
DEFAULT_CFG.TASK.NOISY_POINTGOAL_SENSOR.POSITION_STD = 0.01
DEFAULT_CFG.TASK.NOISY_POINTGOAL_SENSOR.ROTATION_STD = 0.1

DEFAULT_CFG.TASK.EGO_MAP_SENSOR = habitat.Config()
DEFAULT_CFG.TASK.EGO_MAP_SENSOR.TYPE = "EgoMapSensor"
DEFAULT_CFG.TASK.EGO_MAP_SENSOR.METERS_PER_PIXEL = 0.02
DEFAULT_CFG.TASK.EGO_MAP_SENSOR.MARKER_SIZE = 10
DEFAULT_CFG.TASK.EGO_MAP_SENSOR.FOG_OF_WAR = True
DEFAULT_CFG.TASK.EGO_MAP_SENSOR.HFOV = 56
DEFAULT_CFG.TASK.EGO_MAP_SENSOR.VISIBILITY = 2.0


def merge_config(orig_cfg: habitat.Config) -> habitat.Config:
    cfg = DEFAULT_CFG.clone()
    cfg.merge_from_other_cfg(orig_cfg)
    cfg.freeze()
    return cfg


def get_config(cfg_path: Optional[str]=None,
               extra_cfg: Optional[List[str]]=None) -> habitat.Config:
    cfg = habitat.get_config(cfg_path, extra_cfg)
    return merge_config(cfg)
