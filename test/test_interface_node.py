#!/usr/bin/env python

from habitat_sim2real.sims.ros.intf_node   import HabitatInterfaceROSNode
from habitat_sim2real.sims.ros.default_cfg import get_config

import numpy
import cv2
import time


cfg = get_config("configs/locobot_pointnav_real.yaml")
node = HabitatInterfaceROSNode(cfg.SIMULATOR.ROS)
time.sleep(3)
print("Node initialized!")

print("Displaying images")
raw_color, raw_depth = node.get_raw_images()
cv2.imshow("Color", raw_color[:,:,::-1])
cv2.imshow("Depth", raw_depth.astype(numpy.float32) / raw_depth.max())
cv2.waitKey()
cv2.destroyAllWindows()

print("Displaying map")
map_grid, map_cell_size, map_origin_pos, map_origin_rot = node.get_map()
cv2.imshow("Map", map_grid.astype(numpy.float32)[::10,::10] / map_grid.max())
cv2.waitKey()
cv2.destroyAllWindows()

print("Robot pose")
pos, rot = node.get_robot_pose()
print(pos, rot)

print("Querying distance")
d = node.get_distance(pos, (pos[0]+1, pos[1], pos[2]))
print(d)

print("Setting camera tilt")
node.set_camera_tilt(-0.5)
print("Resetting camera tilt")
node.set_camera_tilt(0)

print("WARN!! Moving 20cm forward")
input("Press ENTER to confirm!")
node.move_to_relative(0.2, 0, 0)

print("WARN!! Moving 40cm from original pose")
input("Press ENTER to confirm!")
node.move_to_absolute(pos[0]+0.4, pos[1], 0)

print("WARN!! Turning 90deg left")
input("Press ENTER to confirm!")
node.move_to_relative(0, 0, 1.57)

print("WARN!! Turning 90deg right")
input("Press ENTER to confirm!")
node.move_to_relative(0, 0, -1.57)
