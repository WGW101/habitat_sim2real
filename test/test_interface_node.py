#!/usr/bin/env python

from habitat_sim2real import HabitatInterfaceROSNode
from habitat_sim2real import get_config

import numpy
import cv2
import time


cfg = get_config("configs/locobot_pointnav_citi_sim.yaml")
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
map_grid = node.get_map_grid()
disp = numpy.where(map_grid > -1, 1 - map_grid.astype(numpy.float32) / 100, 0.5)
cv2.imshow("Map", disp)
cv2.waitKey()
cv2.destroyAllWindows()

print("Sampling navigable point")
free_pt = node.sample_free_point()
print(free_pt)

print("Robot pose")
pos, rot = node.get_robot_pose()
print(pos, rot)

print("Querying distances")
d = node.get_distance(pos, (pos[0], pos[1], pos[2]-0.5))
print(d)
d = node.get_distance(pos, free_pt)
print(d)

print("Setting camera tilt")
node.set_camera_tilt(-0.5)
print("Done! Press ENTER to continue")
input()
print("Resetting camera tilt")
node.set_camera_tilt(0)

print("WARN!! Moving 20cm forward")
input("Press ENTER to confirm!")
node.move_to_relative(0.2)

print("WARN!! Moving 40cm from original pose")
input("Press ENTER to confirm!")
node.move_to_absolute([pos[0], pos[1], pos[2]-0.4])

print("WARN!! Turning 90deg left")
input("Press ENTER to confirm!")
node.move_to_relative(0, 1.57)

print("WARN!! Turning 90deg right")
input("Press ENTER to confirm!")
node.move_to_relative(0, -1.57)

print("Test collision! Please hit one of the bumper.")
input("Press ENTER to confirm!")
print("Has collided:", node.has_collided())
print("Clearing collision flag")
node.clear_collided()
print("Has collided:", node.has_collided())
