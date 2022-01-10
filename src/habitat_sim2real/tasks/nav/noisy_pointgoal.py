from typing import Any
import numpy as np
import quaternion

from habitat.core.registry import registry
from habitat.core.simulator import Observations
from habitat.core.dataset import Episode
from habitat.tasks.nav.nav import PointGoalSensor


@registry.register_sensor
class NoisyPointGoalWithGPSAndCompassSensor(PointGoalSensor):
    cls_uuid: str = "pointgoal_with_gps_compass"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

    def get_observation(self, observations: Observations, episode: Episode,
                              *args: Any, **kwargs: Any) -> np.ndarray:
        agent_state = self._sim.get_agent_state()

        pos_noise = np.random.normal(0, self.config.POSITION_STD, (3,))
        agent_position = agent_state.position + pos_noise

        angle_noise = np.random.normal(0, np.radians(self.config.ROTATION_STD), (1,))
        rot_noise = np.quaternion()
        rot_noise.real = np.cos(0.5 * angle_noise)
        rot_noise.imag = np.sin(0.5 * angle_noise) * self._sim.up_vector
        rotation_world_agent = rot_noise * agent_state.rotation

        goal_position = np.array(episode.goals[0].position, dtype=np.float32)

        return self._compute_pointgoal(agent_position, rotation_world_agent, goal_position)

