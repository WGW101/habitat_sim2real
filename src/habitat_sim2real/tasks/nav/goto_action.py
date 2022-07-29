from typing import Any

import numpy as np
import quaternion
from gym import spaces

from habitat.core.registry import registry
from habitat.core.dataset import Episode
from habitat.core.embodied_task import EmbodiedTask, SimulatorTaskAction
from habitat.core.simulator import Observations

import rospy
from visualization_msgs.msg import MarkerArray, Marker
from std_msgs.msg import ColorRGBA
MULTION_COLORS = {
    "cylinder_black": ColorRGBA(r=0.0, g=0.0, b=0.0, a=1.0),
    "cylinder_blue": ColorRGBA(r=0.0, g=0.0, b=1.0, a=1.0),
    "cylinder_cyan": ColorRGBA(r=0.0, g=1.0, b=1.0, a=1.0),
    "cylinder_green": ColorRGBA(r=0.0, g=1.0, b=0.0, a=1.0),
    "cylinder_pink": ColorRGBA(r=1.0, g=0.75, b=0.8, a=1.0),
    "cylinder_red": ColorRGBA(r=1.0, g=0.0, b=0.0, a=1.0),
    "cylinder_white": ColorRGBA(r=1.0, g=1.0, b=1.0, a=1.0),
    "cylinder_yellow": ColorRGBA(r=1.0, g=1.0, b=0.0, a=1.0),
}
def DEBUG_PUBLISH_MARKER(episode):
    arr = MarkerArray()
    for t, stp in enumerate(episode.steps):
        pos = stp.goals[0].position
        mrk = Marker()
        mrk.header.stamp = rospy.Time.now()
        mrk.header.frame_id = "habitat_map"
        mrk.ns = "goals"
        mrk.id = t
        mrk.type = Marker.CYLINDER
        mrk.pose.position.x = pos[0]
        mrk.pose.position.y = pos[1]
        mrk.pose.position.z = pos[2]
        mrk.pose.orientation.x = np.sin(0.25 * np.pi)
        mrk.pose.orientation.w = np.cos(0.25 * np.pi)
        mrk.scale.x = 0.2
        mrk.scale.y = 0.2
        mrk.scale.z = 1.0
        mrk.color = MULTION_COLORS[stp.object_category]
        arr.markers.append(mrk)
    rospy.Publisher("~mark", MarkerArray, queue_size=1, latch=True).publish(arr)


@registry.register_task_action
class GotoAction(SimulatorTaskAction):
    @property
    def action_space(self) -> spaces.Dict:
        return spaces.Dict({"x": spaces.Box(-np.inf, np.inf, (1,), np.float32),
                            "y": spaces.Box(-np.inf, np.inf, (1,), np.float32),
                            "yaw": spaces.Box(-np.pi, np.pi, (1,), np.float32)})

    def reset(self, episode: Episode, task: EmbodiedTask) -> None:
        self.start_pos = np.array(episode.start_position)
        self.start_rot = np.quaternion(episode.start_rotation[3], *episode.start_rotation[:3])
        DEBUG_PUBLISH_MARKER(episode)

    def step(self, x: float, y: float, yaw: float = 0,
             *args: Any, **kwargs: Any) -> Observations:
        src = self._sim.get_agent_state().position
        pos = (self.start_rot * np.quaternion(0, y, 0, -x) * self.start_rot.conj()).vec
        pos += self.start_pos
        rot = self.start_rot * np.quaternion(np.cos(0.5 * yaw), 0, np.sin(0.5 * yaw), 0)
        if self._config.MAX_DISTANCE_LIMIT > 0:
            path = self._sim.get_straight_shortest_path_points(src, pos)
            if not path:
                return self._sim.get_observations_at()
            path = np.array(path)
            remain = self._config.MAX_DISTANCE_LIMIT
            prv = path[0]
            limit = False
            for nxt in path[1:]:
                seg = nxt - prv
                d = np.linalg.norm(seg)
                if d > remain:
                    limit = True
                    break
                remain -= d
                prv = nxt
            if limit:
                pos = prv + seg * remain / d
                yaw = np.arctan2(-seg[0], -seg[2])
                rot = np.quaternion(np.cos(0.5 * yaw), 0, np.sin(0.5 * yaw), 0)
        return self._sim.get_observations_at(pos, rot, True)
