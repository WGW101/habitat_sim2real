from typing import Optional, Any

import numpy as np
import cv2
from gym import spaces

from habitat.core.registry import registry
from habitat.core.simulator import Simulator, Sensor, SensorTypes
from habitat.core.dataset import Episode
from habitat.config.default import Config
from habitat.utils.visualizations import maps, fog_of_war


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
        if self._last_ep_id != episode.episode_id:
            self._topdown_map = maps.get_topdown_map_from_sim(self._sim, meters_per_pixel=mppx)
            self._fog = np.zeros_like(self._topdown_map)

            for goal in episode.goals:
                i, j = maps.to_grid(goal.position[2], goal.position[0],
                                    self._topdown_map.shape, self._sim)
                cv2.circle(self._topdown_map, (j, i), mrk, maps.MAP_TARGET_POINT_INDICATOR, -1)
            self._last_ep_id = episode.episode_id

        s = self._sim.get_agent_state()
        i, j = maps.to_grid(s.position[2], s.position[0], self._topdown_map.shape, self._sim)
        a = 2 * np.arctan(s.rotation.y / s.rotation.w)
        d = self.config.VISIBILITY
        topdown_map = self._topdown_map.copy()
        if self.config.FOG_OF_WAR:
            fov = self.config.HFOV
            self._fog = fog_of_war.reveal_fog_of_war(self._topdown_map, self._fog,
                                                     np.array((i, j)), np.pi + a,
                                                     fov, d / mppx)
            topdown_map[self._fog == 0] = maps.MAP_INVALID_POINT

        h, w = topdown_map.shape
        rot = cv2.getRotationMatrix2D((j, i), -np.degrees(a), 1.0)
        rot_h = int(h * abs(rot[0, 0]) + w * abs(rot[0, 1]))
        rot_w = int(h * abs(rot[0, 1]) + w * abs(rot[0, 0]))
        rot[0, 2] += 0.5 * rot_w - j
        rot[1, 2] += 0.5 * rot_h - i
        rot_map = cv2.warpAffine(topdown_map, rot, (rot_w, rot_h),
                                 borderValue=maps.MAP_INVALID_POINT)
        res = int(2 * d / mppx)
        ego_map = np.zeros((res, res), dtype=np.uint8)
        oi, oj = max(0, (res - rot_h) // 2), max(0, (res - rot_w) // 2)
        y, x = max(0, (rot_h - res) // 2), max(0, (rot_w - res) // 2)
        ego_map[oi:oi + rot_h, oj: oj + rot_w] = rot_map[y:y + res, x: x + res]
        return ego_map

