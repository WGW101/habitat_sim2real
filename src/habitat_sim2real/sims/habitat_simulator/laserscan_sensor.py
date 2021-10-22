from typing import Any, Dict, ClassVar
import math

import numpy as np
import gym

import habitat_sim

from habitat.core.registry import registry
from habitat.core.logging import logger
from habitat.core.simulator import Sensor, SensorTypes, SensorSuite
from habitat.config.default import Config
from habitat.sims.habitat_simulator.habitat_simulator import HabitatSim


@registry.register_sensor
class HabitatSimLaserScanSensor(Sensor):
    cls_uuid: ClassVar[str] = "scan"

    def __init__(self, config: Config, *args: Any, **kwargs: Any) -> None:
        self.config = config
        if any(a != 0 for a in config.ORIENTATION):
            logger.warning("Ignored laser scan orientation config: not currently supported")
        self.angles = np.arange(self.config.MIN_ANGLE, self.config.MAX_ANGLE,
                                self.config.INC_ANGLE)
        self.num_rays, = self.angles.shape
        self._num_depth_cams = math.ceil((self.config.MAX_ANGLE - self.config.MIN_ANGLE)
                                         * 2 / math.pi)
        self._depth_cams_w = math.ceil(0.5 * math.pi / self.config.INC_ANGLE)
        self._depth_rect = np.sqrt(np.linspace(-1, 1, self._depth_cams_w)**2 + 1)
        super().__init__(*args, config=config, **kwargs)

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return HabitatSimLaserScanSensor.cls_uuid

    def _get_sensor_type(self, *args: Any, **kwargs: Any) -> SensorTypes:
        return SensorTypes.DEPTH

    def _get_observation_space(self, *args: Any, **kwargs: Any) -> gym.Space:
        if self.config.POINTS_FORMAT == "POLAR":
            return gym.spaces.Box(np.array([[self.config.MIN_RANGE, self.config.MIN_ANGLE]
                                            for _ in range(self.num_rays)], dtype=np.float32),
                                  np.array([[self.config.MAX_RANGE, self.config.MAX_ANGLE]
                                            for _ in range(self.num_rays)], dtype=np.float32))
        elif self.config.POINTS_FORMAT == "CARTESIAN":
            return gym.spaces.Box(-self.config.MAX_RANGE, self.config.MAX_RANGE,
                                  (self.num_rays, 2), dtype=np.float32)

    def get_observation(self, sim_obs: Dict[str, Any]) -> np.ndarray:
        ranges = np.concatenate([sim_obs[f"scan{i}"][0, ::-1] * self._depth_rect
                                 for i in range(self._num_depth_cams)])
        ranges = np.clip(ranges[:self.num_rays], self.config.MIN_RANGE, self.config.MAX_RANGE)
        if self.config.NOISE_RATIO > 0:
            ranges *= np.random.normal(1, self.config.NOISE_RATIO, self.num_rays)
        if self.config.POINTS_FORMAT == "POLAR":
            return np.stack([ranges, self.angles], 1).astype(np.float32)
        elif self.config.POINTS_FORMAT == "CARTESIAN":
            return np.stack([ranges * np.cos(self.angles),
                             ranges * np.sin(self.angles)], 1).astype(np.float32)


@registry.register_simulator(name="Sim-v1")
class HabitatSim(HabitatSim):
    def create_sim_config(self, _sensor_suite: SensorSuite) -> habitat_sim.Configuration:
        scan_sensor = _sensor_suite.sensors.pop(HabitatSimLaserScanSensor.cls_uuid, None)
        cfg = super().create_sim_config(_sensor_suite)
        if scan_sensor is not None:
            for i in range(scan_sensor._num_depth_cams):
                scan_spec = habitat_sim.SensorSpec()
                scan_spec.uuid = f"scan{i}"
                scan_spec.sensor_type = habitat_sim.SensorType.DEPTH
                scan_spec.position = scan_sensor.config.POSITION
                yaw = np.pi * (0.5 * i + 0.25) + scan_sensor.config.MIN_ANGLE
                scan_spec.orientation = [0, yaw, 0] # TODO: support custom orientation
                scan_spec.resolution = [1, scan_sensor._depth_cams_w]
                scan_spec.parameters["hfov"] = "90"
                cfg.agents[0].sensor_specifications.append(scan_spec)
            _sensor_suite.sensors[HabitatSimLaserScanSensor.cls_uuid] = scan_sensor
        return cfg
