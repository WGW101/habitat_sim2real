#!/usr/bin/env python

import os
os.environ["GLOG_minloglevel"] = "2"
import glob
import random

import habitat
import cv2
import numpy
import quaternion


IMG_DIR_PATH = "out/real/expe_man_ctrl/"
CFG_PATH = "configs/locobot_pointnav_citi_sim.yaml"
N_IMAGES = 1


img_paths = glob.glob(os.path.join(IMG_DIR_PATH, "*_rgb.jpeg"))
random.shuffle(img_paths)
img_paths.insert(0, os.path.join(IMG_DIR_PATH, "x0.000_y0.000_z0.010_r0_rgb.jpeg"))

cfg = habitat.get_config(CFG_PATH)
cfg.defrost()
cfg.SIMULATOR.FORWARD_STEP_SIZE = 0.1
cfg.SIMULATOR.TURN_ANGLE = 5
cfg.SIMULATOR.TILT_ANGLE = 1
cfg.freeze()
sim = habitat.sims.make_sim(cfg.SIMULATOR.TYPE, config=cfg.SIMULATOR)
obs = sim.reset()

offsets = []
for i, path in zip(range(N_IMAGES), img_paths):
    img = cv2.imread(path)
    real_pose = [float(coord_str[1:]) for coord_str in os.path.basename(path).split('_')[:-1]]
    print("Real robot: x {: >7.3f}, y {: >7.3f}, z {: >7.3f}, θ {: >4.0f}".format(*real_pose))
    cv2.imshow("Real robot", img)
    while True:
        st = sim.get_agent_state()
        x, y, z = st.position
        alpha, beta, gamma = quaternion.as_euler_angles(st.rotation)
        if abs(alpha) > 0.1 and abs(gamma) > 0.1:
            beta *= -1
        sim_pose = [x, y, z, numpy.degrees(beta)]
        print("\r Simulator: x {: >7.3f}, y {: >7.3f}, z {: >7.3f}, θ {: >4.0f}".format(
            *sim_pose), end="")
        cv2.imshow("Simulator", obs["rgb"][:, :, ::-1])
        c = cv2.waitKey()
        if c == ord('w'):
            obs = sim.step(1)
        elif c == ord('s'): # BACKWARD
            x = x + cfg.SIMULATOR.FORWARD_STEP_SIZE * numpy.sin(beta)
            z = z + cfg.SIMULATOR.FORWARD_STEP_SIZE * numpy.cos(beta)
            obs = sim.get_observations_at(numpy.array([x, y, z]), st.rotation, True)
        elif c == ord('a'): # STRAFE_LEFT
            x = x - cfg.SIMULATOR.FORWARD_STEP_SIZE * numpy.cos(beta)
            z = z + cfg.SIMULATOR.FORWARD_STEP_SIZE * numpy.sin(beta)
            obs = sim.get_observations_at(numpy.array([x, y, z]), st.rotation, True)
        elif c == ord('d'): # STRAFE_RIGHT
            x = x + cfg.SIMULATOR.FORWARD_STEP_SIZE * numpy.cos(beta)
            z = z - cfg.SIMULATOR.FORWARD_STEP_SIZE * numpy.sin(beta)
            obs = sim.get_observations_at(numpy.array([x, y, z]), st.rotation, True)
        elif c == ord('q'):
            obs = sim.step(2)
        elif c == ord('e'):
            obs = sim.step(3)
        elif c == ord('r'):
            obs = sim.step(4)
        elif c == ord('f'):
            obs = sim.step(5)
        elif c == ord('x'):
            break
    offset = [xs - xr for xs, xr in zip(sim_pose, real_pose)]
    print("\n    Offset: x {: >7.3f}, y {: >7.3f}, z {: >7.3f}, θ {: >4.0f}".format(*offset))
    print("-"*64)
