from typing import Optional, Any, Tuple, List, Dict, NamedTuple

from habitat.core.embodied_task import EmbodiedTask, Measure
from habitat.core.simulator import Simulator, AgentState
from habitat.core.registry import registry

import numpy as np
import quaternion


class MotionErrors(NamedTuple):
    action: str = "START"
    collision: bool = False
    longitudinal: float = 0
    lateral: float = 0
    angular: float = 0


@registry.register_measure
class MotionErrorMeasure(Measure):
    _sim: Simulator
    _task: EmbodiedTask
    _forward_cmd: float
    _turn_cmd: float
    _last_pos: Optional[np.ndarray]
    _last_rot: Optional[np.quaternion]
    _metric: Dict[str, float]

    def __init__(self, *args: Any, sim: Simulator, task: EmbodiedTask,
                       **kwargs: Any) -> None:
        super().__init__(*args, sim=sim, task=task, **kwargs)
        self._sim = sim
        self._task = task
        self._forward_cmd = sim.habitat_config.FORWARD_STEP_SIZE
        self._turn_cmd = sim.habitat_config.TURN_ANGLE
        self._last_pos = None
        self._last_rot = None

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return "motion_error"

    def _get_pos_rot(self) -> Tuple[np.ndarray, np.quaternion]:
        s = self._sim.get_agent_state()
        pos = s.position
        rot = s.rotation if s.rotation.dtype == np.quaternion \
                else np.quaternion(s.rotation[3], *s.rotation[:3])
        return pos, rot

    def reset_metric(self, *args: Any, **kwargs: Any) -> None:
        self._last_pos, self._last_rot = self._get_pos_rot()
        self._metric = MotionErrors()._asdict()

    def update_metric(self, *args: Any, action: Dict[str, Any],
                            **kwargs: Any) -> None:
        action = action["action"]
        action = action if isinstance(action, str) else self._task.get_action_name(action)

        pos, rot = self._get_pos_rot()

        rel_pos = pos - self._last_pos
        rel_pos = (self._last_rot.conjugate() * np.quaternion(0, *rel_pos) * self._last_rot)
        rel_rot = rot * self._last_rot.conjugate()
        rel_rot = np.degrees(2 * np.arctan(rel_rot.y / rel_rot.w)) if rel_rot.w != 0 else 0.0

        if action == "MOVE_FORWARD":
            rel_pos.z += self._forward_cmd
        elif action == "TURN_LEFT":
            rel_rot -= self._turn_cmd
        elif action == "TURN_RIGHT":
            rel_rot += self._turn_cmd

        self._metric = MotionErrors(action, self._sim.previous_step_collided,
                                    -rel_pos.z, -rel_pos.x, rel_rot)._asdict()
        self._last_pos, self._last_rot = pos, rot
