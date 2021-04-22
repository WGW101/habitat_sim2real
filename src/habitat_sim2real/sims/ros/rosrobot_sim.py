import math
import numpy
import cv2
import quaternion
import random

import habitat
from habitat.core.simulator import Simulator, RGBSensor, DepthSensor, SensorSuite
from gym import spaces

from .intf_node import HabitatInterfaceROSNode


class ROSDepthSensor(DepthSensor):
    def _get_observation_space(self, *args, **kwargs):
        return spaces.Box(low=0 if self.config.NORMALIZE_DEPTH else self.config.MIN_DEPTH,
                          high=1 if self.config.NORMALIZE_DEPTH else self.config.MAX_DEPTH,
                          shape=(self.config.HEIGHT, self.config.WIDTH, 1),
                          dtype=numpy.float32)

    def get_observation(self, sim_obs):
        out = cv2.resize(sim_obs[1], (self.config.WIDTH, self.config.HEIGHT))
        out = out.astype(numpy.float32) * 0.001
        out = numpy.clip(out, self.config.MIN_DEPTH, self.config.MAX_DEPTH)
        if self.config.NORMALIZE_DEPTH:
            out = (out - self.config.MIN_DEPTH) \
                  / (self.config.MAX_DEPTH - self.config.MIN_DEPTH)
        return out[:, :, numpy.newaxis]


class ROSRGBSensor(RGBSensor):
    def _get_observation_space(self, *args, **kwargs):
        return spaces.Box(low=0,
                          high=255,
                          shape=(self.config.HEIGHT, self.config.WIDTH, 3),
                          dtype=numpy.uint8)

    def get_observation(self, sim_obs):
        out = cv2.resize(sim_obs[0], (self.config.WIDTH, self.config.HEIGHT))
        return out


class AgentState:
    def __init__(self, p, q):
        self.position = numpy.array([-p[1], p[2], -p[0]])
        self.rotation = quaternion.quaternion(q[3], -q[1], q[2], -q[0])


@habitat.registry.register_simulator(name="ROS-Robot-v0")
class ROSRobot(Simulator):
    def __init__(self, config):
        self.config = config
        self.intf_node = HabitatInterfaceROSNode(self.config.ROS)
        self.cur_camera_tilt = 0
        self._sensor_suite = SensorSuite([ROSRGBSensor(config=self.config.RGB_SENSOR),
                                          ROSDepthSensor(config=self.config.DEPTH_SENSOR)])
        if self.config.ACTION_SPACE_CONFIG == "v0":
            self._action_space = spaces.Discrete(4)
        else: # v1 or pyrobotnoisy
            self._action_space = spaces.Discrete(6)

    @property
    def sensor_suite(self):
        return self._sensor_suite

    @property
    def action_space(self):
        return self._action_space

    def reconfigure(self, config):
        self.config = config

    def reset(self):
        ag_cfg = getattr(self.config, self.config.AGENTS[self.config.DEFAULT_AGENT_ID])
        if ag_cfg.IS_SET_START_STATE:
            pos = numpy.array(ag_cfg.START_POSITION)
            rot = quaternion.quaternion(ag_cfg.START_ROTATION[3], *ag_cfg.START_ROTATION[:3])
            state = self.get_agent_state()
            if not (numpy.allclose(pos, state.position)
                    and quaternion.isclose(rot, state.rotation)):
            self.intf_node.move_to_absolute(-pos[3], -pos[0], 2 * math.atan(rot.y / rot.w))
        self.intf_node.set_camera_tilt(self.config.RGB_SENSOR.ROTATION[0])
        self.intf_node.clear_collided()
        raw_images = self.intf_node.get_raw_images()
        return self._sensor_suite.get_observations(raw_images)

    def step(self, action):
        if action == 0: # STOP
            pass
        elif action == 1: # MOVE_FORWARD
            self.intf_node.move_to_relative(self.config.FORWARD_STEP_SIZE, 0, 0)
        elif action == 2: # TURN_LEFT
            self.intf_node.move_to_relative(0, 0, math.radians(self.config.TURN_ANGLE))
        elif action == 3: # TURN_RIGHT
            self.intf_node.move_to_relative(0, 0, -math.radians(self.config.TURN_ANGLE))
        elif action == 4: # LOOK_UP
            self.cur_camera_tilt -= self.config.TILT_ANGLE
            self.cur_camera_tilt = max(-45, min(self.cur_camera_tilt, 45))
            self.intf_node.set_camera_tilt(math.radians(self.cur_camera_tilt))
        elif action == 5: # LOOK_DOWN
            self.cur_camera_tilt += self.config.TILT_ANGLE
            self.cur_camera_tilt = max(-45, min(self.cur_camera_tilt, 45))
            self.intf_node.set_camera_tilt(math.radians(self.cur_camera_tilt))

        raw_images = self.intf_node.get_raw_images()
        return self._sensor_suite.get_observations(raw_images)

    def get_observations_at(self, position=None, rotation=None, keep_agent_at_new_pose=False):
        if position is None and rotation is None and not keep_agent_at_new_pose:
            raw_images = self.intf_node.get_raw_images()
            return self._sensor_suite.get_observations(raw_images)
        else:
            raise RuntimeError("Can only query observations for current pose on a real robot.")

    def get_agent_state(self, agent_id=0):
        p, q = self.intf_node.get_robot_pose()
        return AgentState(p, q)

    def geodesic_distance(self, pos_a, pos_b, episode=None):
        try:
            iter(pos_b[0])
            all_pos = [pos_a] + pos_b
        except TypeError:
            all_pos = [pos_a, pos_b]
        try:
            return sum(self.intf_node.get_distance((-az, -ax, ay), (-bz, -bx, by))
                       for (ax, ay, az), (bx, by, bz) in zip(all_pos, all_pos[1:]))
        except TypeError: # One of the pos is unreachable
            return None

    def sample_navigable_point(self):
        cur_pos, _ = self.intf_node.get_robot_pose()
        grid, cell_size, origin_pos, origin_rot = self.intf_node.get_map()
        free_pts = list(zip(*numpy.nonzero((0 <= grid) & (grid <= 20))))
        invalid = True
        while invalid:
            y, x = random.choice(free_pts)
            cand = (origin_pos[0] + cell_size * x,
                    origin_pos[1] + cell_size * y,
                    origin_pos[2])
            # Can ROS find a plan to that point?
            invalid = (self.intf_node.get_distance(cur_pos, cand) is None)
        return (-cand[1], cand[2], -cand[0])

    def seed(self, seed):
        random.seed(seed)

    @property
    def up_vector(self):
        return np.array([0.0, 1.0, 0.0])

    @property
    def forward_vector(self):
        return np.array([0.0, 0.0, -1.0])

    def publish_episode_goal(self, goal_pos):
        self.intf_node.publish_episode_goal(-goal_pos[2], -goal_pos[0])
