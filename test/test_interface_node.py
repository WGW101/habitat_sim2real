#!/usr/bin/env python

from habitat_sim2real.sims.ros.intf_node   import HabitatInterfaceROSNode
from habitat_sim2real.sims.ros.default_cfg import get_config

cfg = get_config("configs/locobot_citi_pointnav.yaml")
node = HabitatInterfaceROSNode(cfg.SIMULATOR.ROS)
print("Node initialized!")
