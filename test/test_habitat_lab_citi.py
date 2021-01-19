#!/usr/bin/env python3

import habitat
import cv2

cfg = habitat.get_config("./config/locobot_citi_pointnav.yaml")
sim_cls = habitat.registry.get_simulator(cfg.SIMULATOR.TYPE)
sim = sim_cls(cfg.SIMULATOR)

obs = sim.reset()
while True:
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
