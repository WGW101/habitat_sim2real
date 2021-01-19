#!/usr/bin/env python

from sims.ros.intf_node   import HabitatInterfaceROSNode
from sims.ros.default_cfg import get_config

cfg = get_config("./config/locobot_citi_pointnav.yaml")
node = HabitatInterfaceROSNode(cfg.SIMULATOR.ROS)
