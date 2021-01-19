import math
import numpy
import quaternion
import random

import habitat
from habitat.core.simulator import Simulator, RGBSensor, DepthSensor, SensorSuite
from gym import spaces

from sims.ros.intf_node import HabitatInterfaceROSNode


class ROSDepthSensor(DepthSensor):
    def __init__(self, cfg):
        self.cfg = cfg
        self._obs_space = spaces.Box(low=0 if self.cfg.NORMALIZE_DEPTH else self.cfg.MIN_DEPTH,
                                     high=1 if self.cfg.NORMALIZE_DEPTH else self.cfg.MAX_DEPTH,
                                     shape=(self.cfg.HEIGHT, self.cfg.WIDTH, 1),
                                     dtype=np.float32)

    def _get_observation_space(self):
        return self._obs_space

    def get_observation(self, sim_obs):
        out = sim_obs[1].astype(numpy.float32) / 1000
        out = numpy.clip(raw, self.cfg.MIN_DEPTH, self.cfg.MAX_DEPTH)
        if self.cfg.NORMALIZE_DEPTH:
            out = (out - self.cfg.MIN_DEPTH) / (self.cfg.MAX_DEPTH - self.cfg.MIN_DEPTH)
        return out[:, :, numpy.newaxis]


class ROSRGBSensor(RGBSensor):
    def __init__(self, cfg):
        self.cfg = cfg
        self._obs_space = spaces.Box(low=0,
                                     high=255,
                                     shape=(self.cfg.HEIGHT, self.cfg.WIDTH, 3),
                                     dtype=np.uint8)

    def _get_observation_space(self):
        return self._obs_space

    def get_observation(self, sim_obs):
        return sim_obs[0]


class AgentState:
    def __init__(self, p, q):
        self.position = p
        self.rotation = q


@habitat.registry.register_simulator(name="ROS-Robot-v0")
class ROSRobot(Simulator):
    def __init__(self, cfg):
        self.cfg = cfg
        self.intf_node = HabitatInterfaceROSNode(cfg.ROS)
        self.cur_camera_tilt = 0
        self._sensor_suite = SensorSuite([ROSRGBSensor(cfg),
                                          ROSDepthSensor(cfg)])
        if cfg.ACTION_SPACE_CONFIG == "v0":
            self._action_space = spaces.Discrete(4)
        else: # v1 or pyrobotnoisy
            self._action_space = spaces.Discrete(6)

    @property
    def sensor_suite(self):
        return self._sensor_suite

    @property
    def action_space(self):
        return self._action_space

    def reset(self):
        raw_images = self.intf_node.get_raw_images()
        return self._sensor_suite.get_observations(raw_images)

    def step(self, action):
        if action == 0: # STOP
            pass
        elif action == 1: # MOVE_FORWARD
            self.intf_node.move_to_relative(self.cfg.FORWARD_STEP_SIZE, 0, 0)
        elif action == 2: # TURN_LEFT
            self.intf_node.move_to_relative(0, 0, math.rad(self.cfg.TURN_ANGLE))
        elif action == 3: # TURN_RIGHT
            self.intf_node.move_to_relative(0, 0, -math.rad(self.cfg.TURN_ANGLE))
        elif action == 4: # LOOK_UP
            self.cur_camera_tilt -= self.cfg.TILT_ANGLE
            self.cur_camera_tilt = max(-45, min(self.cur_camera_tilt, 45))
            self.intf_node.set_camera_tilt(math.rad(self.cur_camera_tilt))
        elif action == 5: # LOOK_DOWN
            self.cur_camera_tilt += self.cfg.TILT_ANGLE
            self.cur_camera_tilt = max(-45, min(self.cur_camera_tilt, 45))
            self.intf_node.set_camera_tilt(math.rad(self.cur_camera_tilt))

        raw_images = self.intf_node.get_raw_images()
        return self._sensor_suite.get_observations(raw_images)

    def get_agent_state(self, agent_id=0):
        p, q = self.intf_node.get_robot_pose()
        return AgentState(numpy.array([-p[1], p[2], -p[0]]), quaternion.quaternion(*q))

    def geodesic_distance(self, pos_a, pos_b, episode=None):
        return self.intf_node.get_distance((-pos_a[2], -pos_a[0], pos_a[1]),
                                           (-pos_b[2], -pos_b[0], pos_b[1]))

    def sample_navigable_point(self):
        grid, cell_size, origin_pos, origin_rot = self.intf_node.get_map()
        free_pts = list(zip(*numpy.nonzero((0 <= grid) & (grid <= 20))))
        pt = random.choice(free_pts)
        return (-origin_pos[1] - cell_size * pt[0],
                origin_pos[2],
                -origin_pos[0] - cell_size * pt[1])

    def seed(self, seed):
        random.seed(seed)

