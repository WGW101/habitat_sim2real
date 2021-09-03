from typing import Union, Any, Optional, Dict

from gym.spaces import Space
import numpy as np
import quaternion

from habitat.core.simulator import Simulator, Observations, AgentState
from habitat.config.default import Config
from habitat.sims import make_sim


def make_parallel(master_cls, *, slave_type: Optional[str]=None,
                                 slave_config: Optional[Config]=None):
    if slave_config is None:
        if slave_type is None:
            raise ValueError("Please provide either slave_type or slave_config kwarg")
        slave_config = Config()
    if slave_type is not None:
        slave_config.TYPE = slave_type


    class ParallelSimulator(master_cls):
        _slave_sim: Simulator
        _state_diff: AgentState
        _collision_diff: bool
        _slave_obs: Optional[Observations]

        def __init__(self, master_config: Config, *args: Any, **kwargs: Any) -> None:
            super().__init__(master_config)
            slave_cfg = master_config.clone()
            slave_cfg.merge_from_other_cfg(slave_config)
            self._slave_sim = make_sim(slave_cfg.TYPE, config=slave_cfg)
            self._state_diff = AgentState(None, None)
            self._collision_diff = 0
            self._slave_obs = None

        def _update_slave_state_obs(self) -> None:
            master_s = self.get_agent_state()
            slave_s = self._slave_sim.get_agent_state()
            self._state_diff.position = slave_s.position - master_s.position
            self._state_diff.rotation = slave_s.rotation * master_s.rotation.conjugate()
            self.collision_diff = (self._slave_sim.previous_step_collided
                                   != self.previous_step_collided)
            self.slave_obs = self._slave_sim.get_observations_at(master_s.position,
                                                                 master_s.rotation,
                                                                 True)

        def get_last_state_diff(self) -> AgentState:
            return self._state_diff

        def get_last_collision_diff(self) -> bool:
            return self._collision_diff

        def get_last_slave_obs(self) -> Observations:
            return self._slave_obs

        def reset(self) -> Observations:
            obs = super().reset()
            self._slave_sim.reset()
            self._update_slave_state_obs()
            return obs

        def step(self, action: Union[str, int], *args: Any, **kwargs: Any) -> Observations:
            obs = super().step(action, *args, **kwargs)
            self._slave_sim.step(action, *args, **kwargs)
            self._update_slave_state_obs()
            return obs

        def seed(self, seed: int) -> None:
            super().seed(seed)
            self._slave_sim.seed(seed)

        def reconfigure(self, config: Config) -> None:
            super().reconfigure(config)
            slave_cfg = config.clone()
            slave_cfg.merge_from_other_cfg(slave_config)
            self._slave_sim.reconfigure(slave_cfg)

    return ParallelSimulator
