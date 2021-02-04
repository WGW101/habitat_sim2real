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
N_MEASURES = 5
USAGE = """\
Navigate in the simulation to get the same observation as on the real robot
Controls:
        'forward' ---\    /--- 'turn right'
      'turn left' - q w  e r - 'reset to sim initial pos'      i - 'tilt camera up'
    'strafe left' - a s* d --- 'strafe right'                  k - 'tilt camera down'
  'discard image' --- x  c --- 'confirm measure'
                       *  *  * 'backward'
"""


img_paths = glob.glob(os.path.join(IMG_DIR_PATH, "*_rgb.jpeg"))
random.shuffle(img_paths)
real_origin = os.path.join(IMG_DIR_PATH, "x0.000_y0.000_z0.000_r0_rgb.jpeg")
img_paths.remove(real_origin)
img_paths.insert(0, real_origin)

cfg = habitat.get_config(CFG_PATH)
cfg.defrost()
cfg.SIMULATOR.FORWARD_STEP_SIZE = 0.05
cfg.SIMULATOR.TURN_ANGLE = 1
cfg.SIMULATOR.TILT_ANGLE = 1
cfg.freeze()
sim = habitat.sims.make_sim(cfg.SIMULATOR.TYPE, config=cfg.SIMULATOR)
obs = sim.reset()

print(USAGE)

measure_cnt = 0
offsets = []
for path in img_paths:
    img = cv2.imread(path)
    filename = os.path.basename(path)
    x_real, y_real, z_real, a_deg_real = (float(s[1:]) for s in filename.split('_')[:-1])
    a_real = numpy.radians(a_deg_real)
    print(f"Real robot: x={x_real: >7.3f}, y={y_real: >7.3f}, z={z_real: >7.3f}, θ={a_deg_real: >+6.3f}")
    cv2.imshow("Real robot", img)
    reject = False
    while True:
        st = sim.get_agent_state()
        x_sim, y_sim, z_sim = st.position
        alpha, a_sim, gamma = quaternion.as_euler_angles(st.rotation)
        if abs(alpha) > 1.5 and abs(gamma) > 1.5:
            a_sim *= -1
        a_deg_sim = numpy.degrees(a_sim)
        print(f"\r Simulator: x={x_sim: >7.3f}, y={y_sim: >7.3f}, z={z_sim: >7.3f}, θ={a_deg_sim: >+6.3f}", end="")
        cv2.imshow("Simulator", obs["rgb"][:, :, ::-1])
        c = cv2.waitKey()
        if c == ord('w'):
            obs = sim.step(1)
        elif c == ord('q'):
            obs = sim.step(2)
        elif c == ord('e'):
            obs = sim.step(3)
        elif c == ord('i'):
            obs = sim.step(4)
        elif c == ord('k'):
            obs = sim.step(5)
        elif c == ord('r'):
            obs = sim.reset()
        elif c == ord('s'): # BACKWARD
            x = x_sim + cfg.SIMULATOR.FORWARD_STEP_SIZE * numpy.sin(a_sim)
            z = z_sim + cfg.SIMULATOR.FORWARD_STEP_SIZE * numpy.cos(a_sim)
            obs = sim.get_observations_at(numpy.array([x, y_sim, z]), st.rotation, True)
        elif c == ord('a'): # STRAFE_LEFT
            x = x_sim - cfg.SIMULATOR.FORWARD_STEP_SIZE * numpy.cos(a_sim)
            z = z_sim + cfg.SIMULATOR.FORWARD_STEP_SIZE * numpy.sin(a_sim)
            obs = sim.get_observations_at(numpy.array([x, y_sim, z]), st.rotation, True)
        elif c == ord('d'): # STRAFE_RIGHT
            x = x_sim + cfg.SIMULATOR.FORWARD_STEP_SIZE * numpy.cos(a_sim)
            z = z_sim - cfg.SIMULATOR.FORWARD_STEP_SIZE * numpy.sin(a_sim)
            obs = sim.get_observations_at(numpy.array([x, y_sim, z]), st.rotation, True)
        elif c == ord('x'):
            reject = True
            break
        elif c == ord('c'):
            break
    if reject:
        print()
    else:
        a_off = a_sim - a_real
        x_off = x_sim + x_real * numpy.sin(a_off) + y_real * numpy.cos(a_off)
        y_off = y_sim
        z_off = z_sim + x_real * numpy.cos(a_off) - y_real * numpy.sin(a_off)
        offsets.append([x_off, y_off, z_off, a_off])
        a_deg_off = numpy.degrees(a_off)
        print(f"\n  Offset  : x={x_off: >7.3f}, y={y_off: >7.3f}, z={z_off: >7.3f}, θ={a_deg_off: >4.3f}")
        measure_cnt += 1
        if measure_cnt == N_MEASURES:
            break
    print("-"*64)

cv2.destroyAllWindows()
