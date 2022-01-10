from typing import ClassVar

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
        return spaces.Dict({"x": spaces.Box(-np.inf, np.inf, size=(1,), dtype=np.float32),
                            "z": spaces.Box(-np.inf, np.inf, size=(1,), dtype=np.float32),
                            "yaw": spaces.Box(-np.pi, np.pi, size=(1,), dtype=np.float32)})

    def step(self, x: float, z: float, yaw: float, *args: Any, **kwargs: Any) -> Observations:
        y = self._sim.get_agent_state().position[1]
        target_pos = [x, y, z]
        target_rot = [0, np.sin(0.5 * yaw), 0, np.cos(0.5 * yaw)]
        self._sim.intf_node.move_to_absolute(target_pos, target_rot)
        raw_images = self._sim.intf_node.get_raw_images()
        return self._sim._sensor_suite.get_observations(raw_images)
