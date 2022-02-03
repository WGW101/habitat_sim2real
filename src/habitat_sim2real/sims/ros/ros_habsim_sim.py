from typing import List, Set, Optional, Union

import magnum as mn
from habitat_sim.physics import MotionType

from habitat.core.registry import registry
from habitat.config.default import Config

import rospy
from geometry_msgs.msg import Point
from habitat_sim_ros.srv import GeodesicDistance

from .rosrobot_sim import ROSRobot
from .ros_mngr import ROSManager


class DummyROSTemplateManager:
    ros_mngr: ROSManager
    loaded: Set[str]

    def __init__(self, ros_mngr: ROSManager) -> None:
        self.ros_mngr = ros_mngr
        self.loaded = set()

    def get_file_template_handles(self) -> Set[str]:
        return self.loaded

    def remove_all_templates(self) -> None:
        pass

    def load_configs(self, tmpl_path: str) -> None:
        self.loaded.add(tmpl_path)


class DummyROSSceneNode:
    @property
    def cumulative_bb(self) -> mn.Range3D:
        return mn.Range3D((-0.1, 0.0, -0.1), (0.1, 0.75, 0.1))


class DummyROSObject:
    ros_mngr: ROSManager
    tmpl_id: str
    pos = List[float]
    rot = List[float]

    def __init__(self, ros_mngr: ROSManager, tmpl_id: str) -> None:
        self.ros_mngr = ros_mngr
        self.tmpl_id = tmpl_id
        self.pos = (0.0, 0.0, 0.0)
        self.rot = (0.0, 0.0, 0.0, 1.0)

    @property
    def root_scene_node(self) -> DummyROSSceneNode:
        return DummyROSSceneNode()

    @property
    def object_id(self) -> int:
        return 0

    @property
    def translation(self) -> mn.Vector3:
        return mn.Vector3(self.pos)

    @translation.setter
    def translation(self, vec: mn.Vector3) -> None:
        self.pos = [vec.x, vec.y, vec.z]

    @property
    def rotation(self) -> mn.Quaternion:
        return mn.Quaternion(self.rot[:3], self.rot[3])

    @rotation.setter
    def rotation(self, q: mn.Quaternion) -> None:
        self.rot = [q.vector.x, q.vector.y, q.vector.z, q.scalar]

    @property
    def motion_type(self) -> MotionType:
        return MotionType.STATIC

    @motion_type.setter
    def motion_type(self, t: MotionType) -> None:
        self.ros_mngr.spawn_object(self.tmpl_id, self.pos, self.rot)


class DummyROSObjectManager:
    ros_mngr: ROSManager
    to_clear: bool

    def __init__(self, ros_mngr: ROSManager) -> None:
        self.ros_mngr = ros_mngr
        self.to_clear = False

    def add_object_by_template_handle(self, tmpl_hdl) -> DummyROSObject:
        self.to_clear = True
        return DummyROSObject(self.ros_mngr, tmpl_hdl)

    def remove_object_by_id(self, obj_id) -> None:
        if self.to_clear:
            self.ros_mngr.clear_objects()
            self.to_clear = False


@registry.register_simulator(name="ROS-HabitatSim-v0")
class ROSHabitatSim(ROSRobot):
    ros_mngr: ROSManager
    geo_dist_proxy: rospy.ServiceProxy

    def __init__(self, config: Config) -> None:
        self.ros_mngr = ROSManager(config)
        self.ros_mngr.start()
        self.geo_dist_proxy = rospy.ServiceProxy(
            "/habitat_sim/geodesic_distance", GeodesicDistance
        )
        super().__init__(config)

    def reset(self) -> None:
        self.has_published_goal = False
        raw_obs = self.intf_node.get_raw_observations()
        return self._sensor_suite.get_observations(raw_obs)

    def reconfigure(self, habitat_config: Config) -> None:
        super().reconfigure(habitat_config)
        self.ros_mngr.stop_slam()
        self.ros_mngr.reconfigure_sim(habitat_config)
        self.ros_mngr.restart_slam()

    def close(self) -> None:
        super().close()
        self.ros_mngr.stop()

    def get_rigid_object_manager(self) -> DummyROSObjectManager:
        return DummyROSObjectManager(self.ros_mngr)

    def get_object_template_manager(self) -> DummyROSTemplateManager:
        return DummyROSTemplateManager(self.ros_mngr)

    def geodesic_distance(self,
        src: List[float],
        destinations: Union[List[float], List[List[float]]],
        episode=None
    ) -> float:
        try:
            iter(destinations[0])
        except TypeError:
            destinations = [destinations]
        if not self.has_published_goal:
            self.intf_node.publish_episode_goal(destinations[0])
            self.has_published_goal = True
        return min(self.geo_dist_proxy(
            Point(-src[2], -src[0], src[1]),
            Point(-dst[2], -dst[0], dst[1])
        ).distance for dst in destinations)
