import numpy as np
import quaternion as quat
import cv2
import gym

import habitat
from habitat.core.simulator import (
    Simulator, RGBSensor, DepthSensor, SensorSuite, Sensor, SensorTypes
)
from habitat.utils.visualizations.maps import (
    MAP_INVALID_POINT, MAP_VALID_POINT, MAP_BORDER_INDICATOR
)
from gym import spaces

from .intf_node import HabitatInterfaceROSNode


class ROSDepthSensor(DepthSensor):
    def _get_observation_space(self, *args, **kwargs):
        return spaces.Box(low=0 if self.config.NORMALIZE_DEPTH else self.config.MIN_DEPTH,
                          high=1 if self.config.NORMALIZE_DEPTH else self.config.MAX_DEPTH,
                          shape=(self.config.HEIGHT, self.config.WIDTH, 1),
                          dtype=np.float32)

    def get_observation(self, sim_obs):
        out = cv2.resize(sim_obs[1], (self.config.WIDTH, self.config.HEIGHT))
        out = out.astype(np.float32) * 0.001
        out = np.clip(out, self.config.MIN_DEPTH, self.config.MAX_DEPTH)
        if self.config.NORMALIZE_DEPTH:
            out = (out - self.config.MIN_DEPTH) \
                  / (self.config.MAX_DEPTH - self.config.MIN_DEPTH)
        return out[:, :, np.newaxis]


class ROSRGBSensor(RGBSensor):
    def _get_observation_space(self, *args, **kwargs):
        return spaces.Box(low=0,
                          high=255,
                          shape=(self.config.HEIGHT, self.config.WIDTH, 3),
                          dtype=np.uint8)

    def get_observation(self, sim_obs):
        out = cv2.resize(sim_obs[0], (self.config.WIDTH, self.config.HEIGHT))
        return out


class ROSScanSensor(Sensor):
    def __init__(self, config, *args, **kwargs):
        self.config = config
        self.angles = np.arange(config.MIN_ANGLE, config.MAX_ANGLE, config.INC_ANGLE)
        self.num_rays = self.angles.shape[0]
        super().__init__(*args, config=config, **kwargs)

    def _get_uuid(self, *args, **kwargs):
        return "scan"

    def _get_sensor_type(self, *args, **kwargs):
        return SensorTypes.DEPTH

    def _get_observation_space(self, *args, **kwargs):
        return gym.spaces.Box(np.array([[self.config.MIN_RANGE, self.config.MIN_ANGLE]
                                        for _ in range(self.num_rays)], dtype=np.float32),
                              np.array([[self.config.MAX_RANGE, self.config.MAX_ANGLE]
                                        for _ in range(self.num_rays)], dtype=np.float32),
                              dtype=np.float32)

    def get_observation(self, sim_obs):
        ranges = np.clip(sim_obs[2], self.config.MIN_RANGE, self.config.MAX_RANGE)
        return np.stack((ranges, self.angles), -1)


class AgentState:
    def __init__(self, p, q):
        self.position = np.array(p)
        self.rotation = quat.quaternion(q[3], *q[:3])

    def __repr__(self):
        return "AgentState(position={}, rotation={})".format(self.position, self.rotation)


class DummyROSAgent:
    def __init__(self, state):
        self.state = state


class DummyROSPathfinder:
    def __init__(self, intf_node):
        self._intf_node = intf_node

    def get_bounds(self):
        return self._intf_node.get_map_bounds()

    def get_topdown_view(self, *args, **kwargs):
        thresh = self._intf_node.cfg.MAP_FREE_THRESH
        grid = self._intf_node.get_map_grid()
        topdown = np.full(grid.shape, MAP_INVALID_POINT, dtype=np.uint8)
        topdown[(grid >= 0) & (grid < thresh)] = MAP_VALID_POINT
        topdown[grid >= thresh] = MAP_BORDER_INDICATOR
        return topdown


@habitat.registry.register_simulator(name="ROS-Robot-v0")
class ROSRobot(Simulator):
    def __init__(self, config):
        self.habitat_config = config
        self.intf_node = HabitatInterfaceROSNode(config.ROS)
        self.pathfinder = DummyROSPathfinder(self.intf_node)
        self.cur_camera_tilt = 0
        self._sensor_suite = SensorSuite([ROSRGBSensor(config=config.RGB_SENSOR),
                                          ROSDepthSensor(config=config.DEPTH_SENSOR),
                                          ROSScanSensor(config=config.SCAN_SENSOR)])
        if config.ACTION_SPACE_CONFIG == "v0":
            self._action_space = spaces.Discrete(4)
        elif config.ACTION_SPACE_CONFIG == "v1":
            if self.intf_node.can_tilt_cam:
                self._action_space = spaces.Discrete(6)
            else:
                habitat.logger.warning("Camera tilt control not available. "
                                       + "Falling back to action space config v0. "
                                       + "Actions TILT_UP and TILT_DOWN will be ignored.")
                self._action_space = spaces.Discrete(4)

        self.has_published_goal = False
        self.previous_step_collided = False

    @property
    def sensor_suite(self):
        return self._sensor_suite

    @property
    def action_space(self):
        return self._action_space

    def reconfigure(self, config):
        self.habitat_config = config

    def reset(self):
        self.has_published_goal = False
        ag_cfg = getattr(self.habitat_config,
                         self.habitat_config.AGENTS[self.habitat_config.DEFAULT_AGENT_ID])
        if ag_cfg.IS_SET_START_STATE:
            pos = np.array(ag_cfg.START_POSITION)
            rot = quat.quaternion(ag_cfg.START_ROTATION[3], *ag_cfg.START_ROTATION[:3])
            state = self.get_agent_state()
            if not (np.allclose(pos, state.position)
                    and quat.isclose(rot, state.rotation)):
                self.intf_node.cancel_move_on_bump = False
                self.intf_node.move_to_absolute(ag_cfg.START_POSITION, ag_cfg.START_ROTATION)
                self.intf_node.cancel_move_on_bump = True
        self.intf_node.set_camera_tilt(self.habitat_config.RGB_SENSOR.ORIENTATION[0])
        self.intf_node.clear_collided()
        self.previous_step_collided = False
        raw_obs = self.intf_node.get_raw_observations()
        return self._sensor_suite.get_observations(raw_obs)

    def step(self, action):
        if action == 0: # STOP
            pass
        elif action == 1: # MOVE_FORWARD
            self.intf_node.move_to_relative(self.habitat_config.FORWARD_STEP_SIZE, 0)
        elif action == 2: # TURN_LEFT
            self.intf_node.move_to_relative(0, np.radians(self.habitat_config.TURN_ANGLE))
        elif action == 3: # TURN_RIGHT
            self.intf_node.move_to_relative(0, -np.radians(self.habitat_config.TURN_ANGLE))
        elif action == 4: # LOOK_UP
            self.cur_camera_tilt -= self.habitat_config.TILT_ANGLE
            self.cur_camera_tilt = max(-45, min(self.cur_camera_tilt, 45))
            self.intf_node.set_camera_tilt(np.radians(self.cur_camera_tilt))
        elif action == 5: # LOOK_DOWN
            self.cur_camera_tilt += self.habitat_config.TILT_ANGLE
            self.cur_camera_tilt = max(-45, min(self.cur_camera_tilt, 45))
            self.intf_node.set_camera_tilt(np.radians(self.cur_camera_tilt))

        has_collided = self.intf_node.has_collided()
        if not self.previous_step_collided and has_collided:
            self.intf_node.clear_collided()
            self.previous_step_collided = True

        raw_obs = self.intf_node.get_raw_observations()
        return self._sensor_suite.get_observations(raw_obs)

    def get_observations_at(self, position=None, rotation=None, keep_agent_at_new_pose=False):
        if position is None and rotation is None:
            raw_obs = self.intf_node.get_raw_observations()
            return self._sensor_suite.get_observations(raw_obs)
        else:
            raise RuntimeError("Can only query observations for current pose on a real robot.")

    def get_agent(self, agent_id=0):
        return DummyROSAgent(self.get_agent_state())

    def get_agent_state(self, agent_id=0):
        p, q = self.intf_node.get_robot_pose()
        return AgentState(p, q)

    def geodesic_distance(self, src, destinations, episode=None):
        try:
            iter(destinations[0])
        except TypeError:
            destinations = [destinations]
        # Kinda hacky... Sim has no way to know the goal when a new episode starts
        # (would require episode to be given as arg in reset)
        # But the first call to geodesic_distance is for the distance_to_goal measure...
        if not self.has_published_goal:
            self.intf_node.publish_episode_goal(destinations[0])
            self.has_published_goal = True
        return min(self.intf_node.get_distance(src, dst) for dst in destinations)

    def sample_navigable_point(self):
        return self.intf_node.sample_free_point()

    def seed(self, seed):
        self.intf_node.seed_rng(seed)

    @property
    def up_vector(self):
        return np.array([0.0, 1.0, 0.0])

    @property
    def forward_vector(self):
        return np.array([0.0, 0.0, -1.0])
