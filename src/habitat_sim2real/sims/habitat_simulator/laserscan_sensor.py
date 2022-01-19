from typing import Any, ClassVar, List, Dict
import math

import numpy as np
import quaternion
import gym

import habitat_sim

from habitat.core.registry import registry
from habitat.core.logging import logger
from habitat.core.simulator import Sensor, SensorTypes, SensorSuite
from habitat.config.default import Config
from habitat.sims.habitat_simulator.habitat_simulator import HabitatSim


def _quaternion_to_eulers(q):
    pitch = np.arctan2(2 * (q.w * q.x + q.y * q.z), 1 - 2 * (q.x**2 + q.y**2))
    yaw = np.arcsin(2 * (q.w * q.y - q.z * q.x))
    roll = np.arctan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y**2 + q.z**2))
    return pitch, yaw, roll


@registry.register_sensor
class HabitatSimLaserScanSensor(Sensor):
    cls_uuid: ClassVar[str] = "scan"

    def __init__(self, config: Config, *args: Any, **kwargs: Any) -> None:
        self.config = config
        self.angles = np.arange(self.config.MIN_ANGLE, self.config.MAX_ANGLE,
                                self.config.INC_ANGLE)
        self.num_rays, = self.angles.shape
        self._num_depth_cams = math.ceil((self.config.MAX_ANGLE - self.config.MIN_ANGLE)
                                         * 2 / math.pi)
        self._depth_cams_width = math.ceil(0.5 * math.pi / self.config.INC_ANGLE)
        self._depth_rect = np.sqrt(np.linspace(-1, 1, self._depth_cams_width)**2 + 1)
        super().__init__(config=config, *args, **kwargs)

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return HabitatSimLaserScanSensor.cls_uuid

    def _get_sensor_type(self, *args: Any, **kwargs: Any) -> SensorTypes:
        return SensorTypes.DEPTH

    def _get_observation_space(self, *args: Any, **kwargs: Any) -> gym.Space:
        if self.config.POINTS_FORMAT == "POLAR":
            return gym.spaces.Box(np.array([[self.config.MIN_RANGE, self.config.MIN_ANGLE]
                                            for _ in range(self.num_rays)], dtype=np.float32),
                                  np.array([[self.config.MAX_RANGE, self.config.MAX_ANGLE]
                                            for _ in range(self.num_rays)], dtype=np.float32),
                                  dtype=np.float32)
        elif self.config.POINTS_FORMAT == "CARTESIAN":
            return gym.spaces.Box(-self.config.MAX_RANGE, self.config.MAX_RANGE,
                                  (self.num_rays, 2), dtype=np.float32)

    def get_observation(self, sim_obs: Dict[str, Any]) -> np.ndarray:
        ranges = np.concatenate([sim_obs[f"_{self.uuid}_cam{i}"][0, ::-1] * self._depth_rect
                                 for i in range(self._num_depth_cams)])
        ranges = np.clip(ranges[:self.num_rays], self.config.MIN_RANGE, self.config.MAX_RANGE)
        if self.config.NOISE_RATIO > 0:
            ranges *= np.random.normal(1, self.config.NOISE_RATIO, self.num_rays)
        if self.config.POINTS_FORMAT == "POLAR":
            return np.stack([ranges, self.angles], 1).astype(np.float32)
        elif self.config.POINTS_FORMAT == "CARTESIAN":
            return np.stack([ranges * np.cos(self.angles),
                             ranges * np.sin(self.angles)], 1).astype(np.float32)

    def get_cams_specs(self) -> List[habitat_sim.CameraSensorSpec]:
        q_self = self.relative_rotation
        specs = []
        for i in range(self._num_depth_cams):
            spec = habitat_sim.CameraSensorSpec()
            spec.uuid = f"_{self.uuid}_cam{i}"
            spec.sensor_type = habitat_sim.SensorType.DEPTH
            spec.position = self.config.POSITION
            yaw = np.pi * (0.5 * i + 0.25) + self.config.MIN_ANGLE
            q_yaw = np.quaternion(np.cos(0.5 * yaw), 0, np.sin(0.5 * yaw), 0)
            q_cam = q_self * q_yaw
            spec.orientation = list(_quaternion_to_eulers(q_cam))
            spec.resolution = [1, self._depth_cams_width]
            spec.hfov = 90
            specs.append(spec)
        return specs

    @property
    def relative_position(self):
        return np.array(self.config.POSITION)

    @property
    def relative_rotation(self):
        pitch, yaw, roll = self.config.ORIENTATION
        return (np.quaternion(np.cos(0.5 * roll), 0, 0, np.sin(0.5 * roll))
                * np.quaternion(np.cos(0.5 * yaw), 0, np.sin(0.5 * yaw), 0)
                * np.quaternion(np.cos(0.5 * pitch), np.sin(0.5 * pitch), 0, 0))

    def get_state(self, agent_state: habitat_sim.AgentState) -> habitat_sim.SixDOFPose:
        s = habitat_sim.SixDOFPose()
        s.position = agent_state.position + self.relative_position
        s.rotation = agent_state.rotation * self.relative_rotation
        return s


@registry.register_simulator(name="Sim-v1")
class HabitatSimCustom(HabitatSim):
    def create_sim_config(self, _sensor_suite: SensorSuite) -> habitat_sim.Configuration:
        scan_sensor = _sensor_suite.sensors.pop(HabitatSimLaserScanSensor.cls_uuid, None)
        cfg = super().create_sim_config(_sensor_suite)
        if scan_sensor is not None:
            cfg.agents[0].sensor_specifications.extend(scan_sensor.get_cams_specs())
            _sensor_suite.sensors[HabitatSimLaserScanSensor.cls_uuid] = scan_sensor
        return cfg
