import habitat


DEFAULT_ROS_CFG = habitat.Config()
DEFAULT_ROS_CFG.NODE_NAME = "habitat_interface_node"
DEFAULT_ROS_CFG.COLOR_IMAGE_TOPIC = "/camera/color/image_raw"
DEFAULT_ROS_CFG.DEPTH_IMAGE_TOPIC = "/camera/depth/image_rect_raw"
DEFAULT_ROS_CFG.IMAGE_SYNC_QUEUE_SIZE = 10
DEFAULT_ROS_CFG.MAP_TOPIC = "/map"
DEFAULT_ROS_CFG.TF_ROBOT_FRAME = "base_footprint"
DEFAULT_ROS_CFG.TF_REF_FRAME = "map"
DEFAULT_ROS_CFG.MOVE_BASE_ACTION_SERVER = "/move_base"
DEFAULT_ROS_CFG.MOVE_BASE_PLAN_SERVICE = "/move_base/make_plan"
DEFAULT_ROS_CFG.MOVE_BASE_PLAN_TOL = 0.3
DEFAULT_ROS_CFG.DYNAMIXEL_SERVICE = "/dynamixel_controller/dynamixel_command"
DEFAULT_ROS_CFG.DYNAMIXEL_STATE_TOPIC = "/dynamixel_controller/dynamixel_state"
DEFAULT_ROS_CFG.DYNAMIXEL_TILT_ID = 9
DEFAULT_ROS_CFG.DYNAMIXEL_TILT_TOL = 16
DEFAULT_ROS_CFG.DYNAMIXEL_TIMEOUT = 5.0
DEFAULT_ROS_CFG.CONNECTION_TIMEOUT = 5.0
DEFAULT_ROS_CFG.BUMPER_TOPIC = "/mobile_base/events/bumper"


def merge_ros_config(cfg, ros_cfg=DEFAULT_ROS_CFG):
    cfg.defrost()
    if hasattr(cfg.SIMULATOR, "ROS"):
        default = ros_cfg.clone()
        default.update(cfg.SIMULATOR.ROS)
        cfg.SIMULATOR.ROS = default
    else:
        cfg.SIMULATOR.ROS = ros_cfg.clone()
    cfg.freeze()
    return cfg


def get_config(cfg_path, extra_cfg=None):
    cfg = habitat.get_config(cfg_path, extra_cfg)
    return merge_ros_config(cfg)
