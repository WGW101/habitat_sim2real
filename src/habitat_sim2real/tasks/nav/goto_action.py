from typing import Any

import numpy as np
import quaternion
from gym import spaces

from habitat.core.registry import registry
from habitat.core.dataset import Episode
from habitat.core.embodied_task import EmbodiedTask, SimulatorTaskAction
from habitat.core.simulator import Observations


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

    def step(self, x: float, y: float, yaw: float = 0,
             *args: Any, **kwargs: Any) -> Observations:
        src = self._sim.get_agent_state().position
        pos = (self.start_rot * np.quaternion(0, y, 0, -x) * self.start_rot.conj()).vec
        pos += self.start_pos
        rot = self.start_rot * np.quaternion(np.cos(0.5 * yaw), 0, np.sin(0.5 * yaw), 0)
        if self._config.MAX_DISTANCE_LIMIT > 0:
            path = np.array(self._sim.get_straight_shortest_path_points(src, pos))
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
