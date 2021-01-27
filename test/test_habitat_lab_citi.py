#!/usr/bin/env python3

import habitat
import cv2
import quaternion
import numpy

cfg = habitat.get_config("configs/locobot_pointnav_citi_sim.yaml")
sim_cls = habitat.registry.get_simulator(cfg.SIMULATOR.TYPE)
sim = sim_cls(cfg.SIMULATOR)

obs = sim.reset()
while True:
    state = sim.get_agent_state()
    print("Pos:", state.position)
    _, th, _ = quaternion.as_euler_angles(state.rotation)
    print("Rot:", state.rotation, "(theta: ", int(numpy.rad2deg(th)), ")")
    cv2.imshow("Color", obs["rgb"][:, :, ::-1])
    cv2.imshow("Depth", obs["depth"] / 5)
    c = cv2.waitKey()
    if c == ord("w"):
        obs = sim.step(1)
    elif c == ord("a"):
        obs = sim.step(2)
    elif c == ord("d"):
        obs = sim.step(3)
    elif c == ord("r"):
        obs = sim.step(4)
    elif c == ord("f"):
        obs = sim.step(5)
    else:
        break
cv2.destroyAllWindows()
