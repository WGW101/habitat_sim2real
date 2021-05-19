import numpy as np
import time
from gym import spaces

import habitat
from habitat.core.simulator import Simulator, RGBSensor, SensorSuite

import jetbot


class JetbotRGBSensor(RGBSensor):
    def __init__(self, config):
        super().__init__(config=config)
        self.cam = jetbot.Camera.instance(width=config.WIDTH, height=config.HEIGHT)

    def _get_observation_space(self, *args, **kwargs):
        return spaces.Box(low=0,
                          high=255,
                          shape=(self.config.HEIGHT, self.config.WIDTH, 3),
                          dtype=np.uint8)

    def get_observation(self):
        return self.cam.value[:, :, ::-1] # BGR to RGB


@habitat.registry.register_simulator(name="Jetbot-v0")
class Jetbot(Simulator):
    def __init__(self, config):
        self.config = config
        self._sensor_suite = SensorSuite([JetbotRGBSensor(config=config.RGB_SENSOR)])
        self._action_space = spaces.Discrete(4)
        self.robot = jetbot.Robot()
        self.ctrl_period = 0.1
        self.fwd_speed = 0.1
        self.ang_speed = 1.0
        self.left_motor_coef = 1.0
        self.right_motor_coef = 1.125

    @property
    def sensor_suite(self):
        return self._sensor_suite

    @property
    def action_space(self):
        return self._action_space

    def reset(self):
        return self._sensor_suite.get_observations()

    def step(self, action):
        if action == 0: # STOP
            return self._sensor_suite.get_observations()
        elif action == 1: # MOVE_FORWARD
            left  = self.left_motor_coef * self.fwd_speed
            right = self.right_motor_coef * self.fwd_speed
            dur   = self.config.FORWARD_STEP_SIZE / self.fwd_speed
        elif action == 2: # TURN_LEFT
            left  = -self.left_motor_coef * self.ang_speed
            right = self.right_motor_coef * self.ang_speed
            dur   = np.radians(self.config.TURN_ANGLE) / self.ang_speed
        elif action == 3: # TURN_RIGHT
            left  = self.left_motor_coef * self.ang_speed
            right = -self.right_motor_coef * self.ang_speed
            dur   = np.radians(self.config.TURN_ANGLE) / self.ang_speed
        self.robot.set_motors(left, right)
        time.sleep(dur)
        self.robot.stop()
        return self._sensor_suite.get_observations()

    def get_observations_at(self, position=None, rotation=None, keep_agent_at_new_pose=False):
        if position is None and rotation is None and not keep_agent_at_new_pose:
            return self._sensor_suite.get_observations()
        else:
            raise RuntimeError("Can only query observations for current pose on a real robot.")
