from habitat.core.registry import registry
from habitat.config.default import Config

from .rosrobot_sim import ROSRobot
from .ros_mngr import ROSManager


@registry.register_simulator(name="ROS-HabitatSim-v0")
class ROSHabitatSim(ROSRobot):
    ros_mngr: ROSManager

    def __init__(self, config: Config) -> None:
        self.ros_mngr = ROSManager()
        self.ros_mngr.start(config)
        super().__init__(config)

    def reset(self) -> None:
        self.has_published_goal = False
        raw_obs = self.intf_node.get_raw_observations()
        return self._sensor_suite.get_observations(raw_obs)

    def reconfigure(self, habitat_config: Config) -> None:
        super().reconfigure(habitat_config)
        self.ros_mngr.stop_slam()
        self.ros_mngr.reconfigure_sim(habitat_config)
        self.ros_mngr.restart_slam()

    def close(self) -> None:
        super().close()
        self.ros_mngr.stop()
