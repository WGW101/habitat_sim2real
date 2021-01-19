from habitat.core.registry import registry
from habitat.sims.pyrobot.pyrobot import PyRobot
from gym import spaces
import math


@registry.register_simulator(name="PyRobot-Discrete-Actions-v0")
class PyRobotDiscreteActions(PyRobot):
    def __init__(self, config):
        super().__init__(config.PYROBOT)
        self.sim_config = config.SIMULATOR
        self.current_tilt = 0
        self._action_space = spaces.Discrete(4) # 6 if look_up / look_down, 4 otherwise

    def step(self, action):
        if action == 0: # STOP
            self._robot.base.stop()
        elif action == 1: # FORWARD
            self._robot.base.go_to_relative((self.sim_config.FORWARD_STEP_SIZE, 0, 0))
        elif action == 2: # TURN LEFT
            self._robot.base.go_to_relative((0,0, math.radians(self.sim_config.TURN_ANGLE)))
        elif action == 3: # TURN RIGHT
            self._robot.base.go_to_relative((0,0,-math.radians(self.sim_config.TURN_ANGLE)))
        elif action == 4: # LOOK UP
            self.current_tilt -= self.sim_config.TILT_ANGLE
            self.current_tilt = max(self.current_tilt, -45)
            self._robot.camera.set_tilt(math.radians(self.current_tilt))
        elif action == 5: # LOOK DOWN
            self.current_tilt += self.sim_config.TILT_ANGLE
            self.current_tilt = min(self.current_tilt, 45)
            self._robot.camera.set_tilt(math.radians(self.current_tilt))

        observations = self._sensor_suite.get_observations(
            robot_obs=self.get_robot_observations()
        )
        return observations
