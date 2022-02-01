from typing import Tuple, List, Optional
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
    uuid: str
    launcher: roslaunch.parent.ROSLaunchParent
    sim_node: roslaunch.core.Node
    sim_proc: Optional[roslaunch.pmon.Process] = None
    sim_load_scene: rospy.ServiceProxy
    sim_respawn_agent: rospy.ServiceProxy
    sim_spawn_object: rospy.ServiceProxy
    sim_clear_objects: rospy.ServiceProxy
    slam_node: roslaunch.core.Node
    slam_proc: Optional[roslaunch.pmon.Process] = None
    slam_restarted_event: threading.Event
    nav_node: roslaunch.core.Node
    nav_proc: Optional[roslaunch.pmon.Process] = None
    nav_clear: rospy.ServiceProxy

    def __init__(self) -> None:
        self.uuid = roslaunch.rlutil.get_or_generate_uuid(None, False)
        self.launcher = roslaunch.parent.ROSLaunchParent(
            self.uuid, [], master_logger_level="ERROR", show_summary=False
        )

        self.sim_node = roslaunch.core.Node(
            "habitat_sim_ros", "habitat_sim_node.py", "habitat_sim",
            output="screen"
        )
        self.sim_load_scene = rospy.ServiceProxy("/habitat_sim/load_scene", LoadScene)
        self.sim_respawn_agent = rospy.ServiceProxy("/habitat_sim/respawn_agent", RespawnAgent)
        self.sim_spawn_object = rospy.ServiceProxy("/habitat_sim/spawn_object", SpawnObject)
        self.sim_clear_objects = rospy.ServiceProxy("/habitat_sim/clear_objects", Empty)

        self.slam_node = roslaunch.core.Node(
            "slam_toolbox", "async_slam_toolbox_node", "slam_toolbox",
            remap_args=[("/scan", "/habitat_sim/scan")]
        )
        self.slam_restarted_event = threading.Event()
        rospy.Subscriber("/map", OccupancyGrid, self.on_map)

        self.nav_node = roslaunch.core.Node(
            "move_base", "move_base", "move_base",
            remap_args=[
                ("/scan", "/habitat_sim/scan"),
                ("/cmd_vel", "/habitat_sim/cmd_vel")
            ]
        )
        self.nav_clear = rospy.ServiceProxy("/move_base/clear_costmaps", Empty)

    def set_params(self, sim_cfg: Config) -> None:
        for param_name in (
            "/slam_toolbox/base_frame",
            "/move_base/global_costmap/robot_base_frame",
            "/move_base/local_costmap/robot_base_frame"
        ):
            rospy.set_param(param_name, sim_cfg.ROS.TF_ROBOT_FRAME)
        rospy.set_param("/habitat_sim/sim/scene_id", sim_cfg.SCENE)

    def start(self, sim_cfg: Config) -> None:
        self.launcher.start()
        self.set_params(sim_cfg)

        self.sim_proc, success = self.launcher.runner.launch_node(self.sim_node)
        if not success:
            raise RuntimeError("Failed to launch habitat_sim_ros")

        self.slam_proc, success = self.launcher.runner.launch_node(self.slam_node)
        if not success:
            raise RuntimeError("Failed to launch slam_toolbox")

        self.nav_proc, success = self.launcher.runner.launch_node(self.nav_node)
        if not success:
            raise RuntimeError("Failed to launch move_base")

    def stop(self) -> None:
        self.launcher.shutdown()

    def on_map(self, map_msg) -> None:
        self.slam_restarted_event.set()

    def stop_slam(self) -> None:
        self.slam_proc.stop()

    def restart_slam(self) -> None:
        self.slam_restarted_event.clear()
        self.slam_proc, success = self.launcher.runner.launch_node(self.slam_node)
        if not success or not self.slam_restarted_event.wait(timeout=30.0):
            raise RuntimeError("Failed to restart slam_toolbox")
        self.nav_clear()

    @staticmethod
    def _make_pose(position: List[float], rotation: List[float]) -> Pose:
        p = Pose()
        p.position.x = -position[2]
        p.position.y = -position[0]
        p.position.z = position[1]
        p.orientation.z = rotation[1]
        p.orientation.w = rotation[3]
        return p

    def reconfigure_sim(self, sim_cfg: Config) -> None:
        self.sim_load_scene(sim_cfg.SCENE)
        if sim_cfg.AGENT_0.IS_SET_START_STATE:
            self.sim_respawn_agent(ROSManager._make_pose(
                sim_cfg.AGENT_0.START_POSITION, sim_cfg.AGENT_0.START_ROTATION
            ))

    def spawn_object(self,
        tmpl_id: str,
        position: Tuple[float, float, float],
        rotation: Tuple[float, float, float, float],
    ) -> None:
        p = Pose()
        p.position.x = -position[2]
        p.position.y = -position[0]
        p.position.z = position[1]
        p.orientation.x = -rotation[2]
        p.orientation.y = -rotation[0]
        p.orientation.z = rotation[1]
        p.orientation.w = rotation[3]
        self.sim_spawn_object(tmpl_id, p)

    def clear_objects(self):
        self.sim_clear_objects()
