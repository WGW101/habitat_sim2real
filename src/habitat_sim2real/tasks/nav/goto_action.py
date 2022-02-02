from typing import Any

import numpy as np
import quaternion
from gym import spaces

from habitat.core.registry import registry
from habitat.core.embodied_task import SimulatorTaskAction
from habitat.core.simulator import Observations


@registry.register_task_action
class ROSGotoAction(SimulatorTaskAction):
    @property
    def action_space(self) -> spaces.Dict:
        return spaces.Dict({"x": spaces.Box(-np.inf, np.inf, (1,), np.float32),
                            "y": spaces.Box(-np.inf, np.inf, (1,), np.float32),
                            "yaw": spaces.Box(-np.pi, np.pi, (1,), np.float32)})

    def step(self, x: float, y: float, yaw: float = 0,
             *args: Any, **kwargs: Any) -> Observations:
        src = self._sim.get_agent_state().position
        target_pos = [y, src[1], -x]
        target_rot = [0, np.sin(0.5 * yaw), 0, np.cos(0.5 * yaw)]
        if self._config.MAX_DISTANCE_LIMIT > 0:
            shortest_path = self._sim.intf_node.get_shortest_path(src, target_pos)
            cumul = 0
            prv = shortest_path[0]
            for dst in shortest_path[1:]:
                cumul += np.sqrt(((dst - prv)**2).sum())
                if cumul >= self._config.MAX_DISTANCE_LIMIT:
                    break
            self._sim.intf_node.move_to_absolute(dst, target_rot)
        else:
            self._sim.intf_node.move_to_absolute(target_pos, target_rot)
        raw_images = self._sim.intf_node.get_raw_images()
        return self._sim._sensor_suite.get_observations(raw_images)
