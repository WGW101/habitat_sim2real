from typing import Union, Any, Optional, Dict, Tuple
import multiprocessing as mp
import enum
import logging
import os

import numpy as np
import quaternion

from habitat.core.simulator import Simulator, Observations, AgentState
from habitat.config.default import Config
from habitat.sims import make_sim


_logger = logging.getLogger(__name__)
_logger.addHandler(logging.StreamHandler())
_logger.setLevel(logging.DEBUG)


class SlaveProcess(mp.Process):
    @staticmethod
    def state_to_tuple(s: AgentState) -> Tuple[float]:
        return (*s.position.tolist(), s.rotation.w, s.rotation.x, s.rotation.y, s.rotation.z)

    @staticmethod
    def tuple_to_state(tup: Tuple[float]) -> AgentState:
        return AgentState(np.array(tup[:3]), np.quaternion(*tup[3:]))

    @staticmethod
    def array_to_shared(src: np.ndarray, dest: mp.Array, typ: str) -> None:
        wrap = np.frombuffer(dest.get_obj(), dtype=typ)
        with dest:
            wrap[:] = src.flatten()

    @staticmethod
    def shared_to_array(src: mp.Array, typ: str) -> np.ndarray:
        wrap = np.frombuffer(src.get_obj(), dtype=typ)
        with src:
            cp = wrap.copy()
        return cp

    class Messages(enum.Enum):
        READY = enum.auto()
        SEED = enum.auto()
        RECONFIGURE = enum.auto()
        GET_STATE = enum.auto()
        GET_COLLIDED = enum.auto()
        GET_OBS_AT = enum.auto()
        RESET = enum.auto()
        STEP = enum.auto()
        RETURN_NONE = enum.auto()
        RETURN_STATE = enum.auto()
        RETURN_COLLIDED = enum.auto()
        RETURN_OBS = enum.auto()
        TERMINATE = enum.auto()

    _config: Config
    _conn: mp.connection.Connection
    shared_rgb: mp.Array
    shared_depth: mp.Array

    def __init__(self, config: Config) -> None:
        super().__init__()
        self._config = config
        self._conn = None
        self.shared_rgb = mp.Array('B', config.RGB_SENSOR.WIDTH * config.RGB_SENSOR.HEIGHT * 3)
        self.shared_depth = mp.Array('f', config.DEPTH_SENSOR.WIDTH * config.DEPTH_SENSOR.HEIGHT)
        _logger.debug("[SLAVE] Initialized slave proc.")

    def connect(self) -> mp.connection.Connection:
        parent_conn, self._conn = mp.Pipe()
        return parent_conn

    def run(self) -> None:
        _logger.debug("[SLAVE] Running slave proc.")
        _logger.debug(f"[SLAVE] pid={os.getpid()}")
        with make_sim(self._config.TYPE, config=self._config) as sim:
            _logger.debug(f"[SLAVE] Created slave sim")
            self._conn.send((SlaveProcess.Messages.READY, None))
            while True:
                _logger.debug(f"[SLAVE] Waiting for msg")
                msg, arg = self._conn.recv()
                if msg == SlaveProcess.Messages.RESET:
                    sim.reset()
                    self._conn.send((SlaveProcess.Messages.RETURN_NONE, None))
                elif msg == SlaveProcess.Messages.SEED:
                    sim.seed(arg)
                    self._conn.send((SlaveProcess.Messages.RETURN_NONE, None))
                elif msg == SlaveProcess.Messages.RECONFIGURE:
                    sim.reconfigure(arg)
                    self._conn.send((SlaveProcess.Messages.RETURN_NONE, None))
                elif msg == SlaveProcess.Messages.GET_STATE:
                    s = sim.get_agent_state()
                    self._conn.send((SlaveProcess.Messages.RETURN_STATE,
                                     SlaveProcess.state_to_tuple(s)))
                elif msg == SlaveProcess.Messages.GET_COLLIDED:
                    self._conn.send((SlaveProcess.Messages.RETURN_COLLIDED,
                                     sim.previous_step_collided))
                elif msg == SlaveProcess.Messages.GET_OBS_AT:
                    s = SlaveProcess.tuple_to_state(arg)
                    obs = sim.get_observations_at(s.position, s.rotation, True)
                    SlaveProcess.array_to_shared(obs['rgb'], self.shared_rgb, 'B')
                    SlaveProcess.array_to_shared(obs['depth'], self.shared_depth, 'f')
                    self._conn.send((SlaveProcess.Messages.RETURN_OBS, None))
                elif msg == SlaveProcess.Messages.STEP:
                    sim.step(arg)
                    self._conn.send((SlaveProcess.Messages.RETURN_NONE, None))
                elif msg == SlaveProcess.Messages.TERMINATE:
                    break

    def get_shared_obs(self) -> Dict[str, np.ndarray]:
        return {"rgb": SlaveProcess.shared_to_array(self.shared_rgb, 'B'),
                "depth": SlaveProcess.shared_to_array(self.shared_depth, 'f')}


def make_sidechain(master_cls, *, slave_type: Optional[str]=None,
                                  slave_config: Optional[Config]=None):
    if slave_config is None:
        if slave_type is None:
            raise ValueError("Please provide either slave_type or slave_config kwarg")
        slave_config = Config()
    if slave_type is not None:
        slave_config.TYPE = slave_type


    class SidechainSimulator(master_cls):
        _slave_proc: SlaveProcess
        _conn: mp.connection.Connection
        _state_diff: AgentState
        _collision_diff: bool
        _slave_obs: Optional[Observations]

        def __init__(self, master_config: Config, *args: Any, **kwargs: Any) -> None:
            super().__init__(master_config)
            _logger.debug("[MASTER] Initialized super() on master proc.")
            _logger.debug(f"[MASTER] pid={os.getpid()}")
            slave_cfg = master_config.clone()
            slave_cfg.merge_from_other_cfg(slave_config)
            mp.set_start_method("forkserver")
            self._slave_proc = SlaveProcess(slave_cfg)
            self._conn = self._slave_proc.connect()
            self._slave_proc.start()

            msg, arg = self._conn.recv()
            assert msg == SlaveProcess.Messages.READY

            self._state_diff = AgentState(None, None)
            self._collision_diff = 0
            self._slave_obs = None
            _logger.debug("[MASTER] Initialized master proc.")

        def _update_slave_state_obs(self) -> None:
            s = self.get_agent_state()

            self._conn.send((SlaveProcess.Messages.GET_STATE, None))
            msg, arg = self._conn.recv()
            assert msg == SlaveProcess.Messages.RETURN_STATE
            self._state_diff.position = np.array(arg[:3]) - s.position
            self._state_diff.rotation = np.quaternion(*arg[3:]) * s.rotation.conjugate()

            self._conn.send((SlaveProcess.Messages.GET_COLLIDED, None))
            msg, arg = self._conn.recv()
            assert msg == SlaveProcess.Messages.RETURN_COLLIDED
            self.collision_diff = (arg != self.previous_step_collided)

            self._conn.send((SlaveProcess.Messages.GET_OBS_AT, SlaveProcess.state_to_tuple(s)))
            msg, arg = self._conn.recv()
            assert msg == SlaveProcess.Messages.RETURN_OBS
            self._slave_obs = self._slave_proc.get_shared_obs()

        def get_last_state_diff(self) -> AgentState:
            return self._state_diff

        def get_last_collision_diff(self) -> bool:
            return self._collision_diff

        def get_last_slave_obs(self) -> Observations:
            return self._slave_obs

        def reset(self) -> Observations:
            obs = super().reset()

            self._conn.send((SlaveProcess.Messages.RESET, None))
            msg, arg = self._conn.recv()
            assert msg == SlaveProcess.Messages.RETURN_NONE

            self._update_slave_state_obs()
            return obs

        def step(self, action: Union[str, int], *args: Any, **kwargs: Any) -> Observations:
            obs = super().step(action, *args, **kwargs)

            self._conn.send((SlaveProcess.Messages.STEP, action))
            msg, arg = self._conn.recv()
            assert msg == SlaveProcess.Messages.RETURN_NONE

            self._update_slave_state_obs()
            return obs

        def seed(self, seed: int) -> None:
            super().seed(seed)

            self._conn.send((SlaveProcess.Messages.SEED, seed))
            msg, arg = self._conn.recv()
            assert msg == SlaveProcess.Messages.RETURN_NONE

        def reconfigure(self, config: Config) -> None:
            super().reconfigure(config)
            slave_cfg = config.clone()
            slave_cfg.merge_from_other_cfg(slave_config)

            self._conn.send((SlaveProcess.Messages.RECONFIGURE, slave_cfg))
            msg, arg = self._conn.recv()
            assert msg == SlaveProcess.Messages.RETURN_NONE

        def close(self) -> None:
            self._conn.send((SlaveProcess.Messages.TERMINATE, None))
            self._slave_proc.join()
            super().close()

    return SidechainSimulator
