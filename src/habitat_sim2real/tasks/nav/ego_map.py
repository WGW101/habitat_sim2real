from typing import Optional, Any

import numpy as np
import cv2
from gym import spaces

from habitat.core.registry import registry
from habitat.core.simulator import Simulator, Sensor, SensorTypes
from habitat.core.dataset import Episode
from habitat.config.default import Config
from habitat.utils.visualizations import maps, fog_of_war


import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
def DEBUG_PUBLISH_MAP_IMG(ego_map):
    rospy.Publisher("~ego_map", Image, queue_size=2, latch=True).publish(
        CvBridge().cv2_to_imgmsg(ego_map)
    )



@registry.register_sensor
class EgoMapSensor(Sensor):
    _sim: Simulator
    _last_ep_id: Optional[str]
    _topdown_map: Optional[np.ndarray]
    _fog: Optional[np.ndarray]

    def __init__(self, sim: Simulator, config: Config, *args: Any, **kwargs: Any) -> None:
        self._sim = sim
        self._last_ep_id = None
        self._topdown_map = None
        self._fog = None
        super().__init__(*args, sim=sim, config=config, **kwargs)

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return "ego_map"

    def _get_sensor_type(self, *args: Any, **kwargs: Any) -> SensorTypes:
        return SensorTypes.PATH

    def _get_observation_space(self, *args: Any, **kwargs: Any) -> spaces.Box:
        res = int(2 * self.config.VISIBILITY / self.config.METERS_PER_PIXEL)
        return spaces.Box(0, 255, (res, res), dtype=np.uint8)

    def get_observation(self, episode: Episode, *args: Any, **kwargs: Any) -> np.ndarray:
        mrk = self.config.MARKER_SIZE // 2
        mppx = self.config.METERS_PER_PIXEL
        if self.config.ONLINE_MAP or self._last_ep_id != episode.episode_id:
            self._topdown_map = maps.get_topdown_map_from_sim(
                self._sim, draw_border=False, meters_per_pixel=mppx
            )
            for goal in episode.goals:
                i, j = maps.to_grid(goal.position[2], goal.position[0],
                                    self._topdown_map.shape, self._sim)
                if self._topdown_map[i, j] != maps.MAP_INVALID_POINT:
                    cv2.circle(self._topdown_map, (j, i), mrk,
                               maps.MAP_TARGET_POINT_INDICATOR, -1)
        if self._last_ep_id != episode.episode_id:
            self._fog = np.zeros(self._topdown_map.shape, dtype=np.bool_)
            self._last_ep_id = episode.episode_id

        s = self._sim.get_agent_state()
        i, j = maps.to_grid(s.position[2], s.position[0], self._topdown_map.shape, self._sim)
        a = 2 * np.arctan(s.rotation.y / s.rotation.w)
        d = self.config.VISIBILITY
        topdown_map = self._topdown_map.copy()
        if self.config.FOG_OF_WAR:
            self._fog = fog_of_war.reveal_fog_of_war(
                self._topdown_map,
                self._fog,
                np.array((i, j)),
                np.pi + a,
                self.config.HFOV,
                d / mppx
            )
            topdown_map[self._fog == 0] = maps.MAP_INVALID_POINT

        h, w = topdown_map.shape
        rot = cv2.getRotationMatrix2D((j, i), -np.degrees(a), 1.0)
        res = int(2 * d / mppx)
        rot[0, 2] += 0.5 * res - j
        rot[1, 2] += 0.5 * res - i
        ego_map = cv2.warpAffine(topdown_map, rot, (res, res), borderValue=maps.MAP_INVALID_POINT)
        DEBUG_PUBLISH_MAP_IMG(ego_map)
        return ego_map
