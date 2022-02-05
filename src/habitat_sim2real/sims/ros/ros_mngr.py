from typing import List, Optional
import math
import threading

import rospy
from geometry_msgs.msg import Pose
from nav_msgs.msg import OccupancyGrid
from habitat_sim_ros.srv import LoadScene, RespawnAgent, SpawnObject
from std_srvs.srv import Empty
import roslaunch

from habitat.config.default import Config
from habitat.tasks.nav.spawned_objectnav import SpawnedObjectGoal


class ROSManager:
    _sim_cfg: Config
    _uuid: str
    _launcher: roslaunch.parent.ROSLaunchParent
    _set_logger_level: rospy.ServiceProxy
    _sim_node: roslaunch.core.Node
    _sim_proc: Optional[roslaunch.pmon.Process] = None
    _sim_load_scene: rospy.ServiceProxy
    _sim_respawn_agent: rospy.ServiceProxy
    _sim_spawn_object: rospy.ServiceProxy
    _sim_clear_objects: rospy.ServiceProxy
    _slam_node: roslaunch.core.Node
    _slam_proc: Optional[roslaunch.pmon.Process] = None
    _slam_restarted_event: threading.Event
    _nav_node: roslaunch.core.Node
    _nav_proc: Optional[roslaunch.pmon.Process] = None
    _nav_clear: rospy.ServiceProxy

    def __init__(self, sim_cfg: Config) -> None:
        self._sim_cfg = sim_cfg.clone()
        self._uuid = roslaunch.rlutil.get_or_generate_uuid(None, False)
        self._launcher = roslaunch.parent.ROSLaunchParent(
            self._uuid, [], master_logger_level="ERROR", show_summary=False
        )

        self._sim_node = roslaunch.core.Node(
            "habitat_sim_ros", "habitat_sim_node.py", "habitat_sim",
            output="screen",
            remap_args=[
                ("/habitat_sim/scan", sim_cfg.ROS.SCAN_TOPIC),
                ("/habitat_sim/map", sim_cfg.ROS.MAP_TOPIC),
                ("/habitat_sim/camera/color/image_raw", sim_cfg.ROS.COLOR_IMAGE_TOPIC),
                ("/habitat_sim/camera/depth/image_raw", sim_cfg.ROS.DEPTH_IMAGE_TOPIC),
            ]
        )
        self._sim_load_scene = rospy.ServiceProxy("/habitat_sim/load_scene", LoadScene)
        self._sim_respawn_agent = rospy.ServiceProxy(
            "/habitat_sim/respawn_agent", RespawnAgent
        )
        self._sim_spawn_object = rospy.ServiceProxy("/habitat_sim/spawn_object", SpawnObject)
        self._sim_clear_objects = rospy.ServiceProxy("/habitat_sim/clear_objects", Empty)

        self._slam_node = roslaunch.core.Node(
            "slam_toolbox", "async_slam_toolbox_node", "slam_toolbox",
            remap_args=[
                ("/scan", sim_cfg.ROS.SCAN_TOPIC),
                ("/map", sim_cfg.ROS.MAP_TOPIC),
            ]
        )
        self._slam_restarted_event = threading.Event()
        rospy.Subscriber(sim_cfg.ROS.MAP_TOPIC, OccupancyGrid, self._on_map)

        self._nav_node = roslaunch.core.Node(
            "move_base", "move_base", "move_base",
            remap_args=[
                ("/scan", sim_cfg.ROS.SCAN_TOPIC),
                ("/cmd_vel", "/habitat_sim/cmd_vel")
            ]
        )
        self._nav_clear = rospy.ServiceProxy("/move_base/clear_costmaps", Empty)

    def _set_params(self) -> None:
        self._set_sim_params(self._sim_cfg)
        self._set_movebase_params()

    def _set_sim_params(self, sim_cfg: Config, ns: str = "/habitat_sim") -> None:
        #rospy.set_param("/use_sim_time", True)
        rospy.set_param(f"{ns}/sim/allow_sliding", sim_cfg.HABITAT_SIM_V0.ALLOW_SLIDING)
        rospy.set_param(f"{ns}/sim/scene_id", sim_cfg.SCENE)
        rospy.set_param(f"{ns}/sim/seed", sim_cfg.SEED)
        self._set_agent_params(sim_cfg.AGENT_0, f"{ns}/agent")
        self._set_sensor_params(sim_cfg.RGB_SENSOR, f"{ns}/color")
        self._set_sensor_params(sim_cfg.DEPTH_SENSOR, f"{ns}/depth")

    def _set_agent_params(self, ag_cfg: Config, ns: str = "/habitat_sim/agent") -> None:
        rospy.set_param(f"{ns}/start_position/random", not ag_cfg.IS_SET_START_STATE)
        rospy.set_param(f"{ns}/start_position/x", -ag_cfg.START_POSITION[2])
        rospy.set_param(f"{ns}/start_position/y", -ag_cfg.START_POSITION[0])
        rospy.set_param(f"{ns}/start_position/z", ag_cfg.START_POSITION[1])
        rospy.set_param(f"{ns}/start_orientation/random", not ag_cfg.IS_SET_START_STATE)
        yaw = math.degrees(
            2 * math.atan(ag_cfg.START_ROTATION[1] / ag_cfg.START_ROTATION[3])
        ) if ag_cfg.START_ROTATION[3] != 0 else 180
        rospy.set_param(f"{ns}/start_orientation/yaw", yaw)
        rospy.set_param(f"{ns}/height", ag_cfg.HEIGHT)
        rospy.set_param(f"{ns}/radius", ag_cfg.RADIUS)

    def _set_sensor_params(self, sensor_cfg: Config, ns: str = "/habitat_sim/color") -> None:
        rospy.set_param(f"{ns}/position/x", -sensor_cfg.POSITION[2])
        rospy.set_param(f"{ns}/position/y", -sensor_cfg.POSITION[0])
        rospy.set_param(f"{ns}/position/z", sensor_cfg.POSITION[1])
        rospy.set_param(f"{ns}/orientation/tilt", -sensor_cfg.ORIENTATION[0])
        rospy.set_param(f"{ns}/height", sensor_cfg.HEIGHT)
        rospy.set_param(f"{ns}/width", sensor_cfg.WIDTH)
        rospy.set_param(f"{ns}/hfov", sensor_cfg.HFOV)

    def _set_movebase_params(self, ns: str = "/move_base") -> None:
        rospy.set_param(f"{ns}/recovery_behavior_enabled", False)
        rospy.set_param(f"{ns}/clearing_rotation_allowed", False)
        rospy.set_param(f"{ns}/NavfnROS/allow_unknown", True)
        self._set_global_costmap_params(f"{ns}/global_costmap")
        self._set_local_costmap_params(f"{ns}/local_costmap")

    def _set_global_costmap_params(self, ns: str = "/move_base/global_costmap") -> None:
        rospy.set_param(f"{ns}/robot_base_frame", "base_footprint")
        rospy.set_param(f"{ns}/robot_base_frame", "base_footprint")
        rospy.set_param(f"{ns}/static_map", True)
        rospy.set_param(f"{ns}/update_frequency", 1.0)
        rospy.set_param(f"{ns}/robot_radius", 0.18)
        rospy.set_param(f"{ns}/inflation_radius", 0.5)

    def _set_local_costmap_params(self, ns: str = "/move_base/local_costmap") -> None:
        rospy.set_param(f"{ns}/robot_base_frame", "base_footprint")
        rospy.set_param(f"{ns}/static_map", False)
        rospy.set_param(f"{ns}/rolling_window" , True)
        rospy.set_param(f"{ns}/width", 3.0)
        rospy.set_param(f"{ns}/height", 3.0)

    def start(self) -> None:
        self._launcher.start()
        self._set_params()

        self._sim_proc, success = self._launcher.runner.launch_node(self._sim_node)
        if not success:
            raise RuntimeError("Failed to launch habitat_sim_ros")

        self._slam_proc, success = self._launcher.runner.launch_node(self._slam_node)
        if not success:
            raise RuntimeError("Failed to launch slam_toolbox")

        self._nav_proc, success = self._launcher.runner.launch_node(self._nav_node)
        if not success:
            raise RuntimeError("Failed to launch move_base")

    def stop(self) -> None:
        self._launcher.shutdown()

    def _on_map(self, map_msg) -> None:
        self._slam_restarted_event.set()

    def stop_slam(self) -> None:
        self._slam_proc.stop()

    def restart_slam(self) -> None:
        self._slam_restarted_event.clear()
        self._slam_proc, success = self._launcher.runner.launch_node(self._slam_node)
        if not success or not self._slam_restarted_event.wait(timeout=30.0):
            raise RuntimeError("Failed to restart slam_toolbox")
        self._nav_clear()

    @staticmethod
    def _make_pose(position: List[float], rotation: List[float]) -> Pose:
        p = Pose()
        p.position.x = -position[2]
        p.position.y = -position[0]
        p.position.z = position[1]
        p.orientation.x = -rotation[2]
        p.orientation.y = -rotation[0]
        p.orientation.z = rotation[1]
        p.orientation.w = rotation[3]
        return p

    def reconfigure_sim(self, sim_cfg: Config) -> None:
        if self._sim_cfg.SCENE != sim_cfg.SCENE:
            self._sim_load_scene(sim_cfg.SCENE)
        if sim_cfg.AGENT_0.IS_SET_START_STATE:
            self._sim_respawn_agent(ROSManager._make_pose(
                sim_cfg.AGENT_0.START_POSITION, sim_cfg.AGENT_0.START_ROTATION
            ))
        self._sim_cfg = sim_cfg.clone()

    def spawn_object(self,
        tmpl_id: str,
        position: List[float],
        rotation: List[float],
    ) -> None:
        self._sim_spawn_object(tmpl_id, ROSManager._make_pose(position, rotation))

    def clear_objects(self):
        self._sim_clear_objects()
