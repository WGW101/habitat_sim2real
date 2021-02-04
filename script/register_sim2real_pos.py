#!/usr/bin/env python

import os
os.environ["GLOG_minloglevel"] = "2"
import glob
import random
import argparse

import habitat
import cv2
import numpy
import quaternion


IN_DIR = "out/real"
CFG_PATH = "configs/locobot_pointnav_citi_sim.yaml"
N_MEASURES = 5
STEP_SIZE = 0.01
TURN_ANGLE = 1
TILT_ANGLE = 1

USAGE = """\
Navigate in the simulation to get the same observation as on the real robot
Controls:
        'forward' ---\    /--- 'turn right'
      'turn left' - q w  e r - 'reset to sim initial pos'      i - 'tilt camera up'
    'strafe left' - a s* d --- 'strafe right'                  k - 'tilt camera down'
  'discard image' --- x  c --- 'confirm measure'
                       *  *  * 'backward'
"""


def parse_args():
    parser = argparse.ArgumentParser(description=USAGE,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-dir", "-i", default=IN_DIR)
    parser.add_argument("--output-dir", "-o", default=None)
    parser.add_argument("--cfg-path", "-c", default=CFG_PATH)
    parser.add_argument("--n-measures", "-n", type=int, default=N_MEASURES)
    parser.add_argument("--step-size", type=float, default=STEP_SIZE)
    parser.add_argument("--turn-angle", type=float, default=TURN_ANGLE)
    parser.add_argument("--tilt-angle", type=float, default=TILT_ANGLE)
    return parser.parse_args()


def setup_sim(step_size, turn_ang, tilt_ang):
    cfg = habitat.get_config(CFG_PATH)
    cfg.defrost()
    cfg.SIMULATOR.FORWARD_STEP_SIZE = step_size
    cfg.SIMULATOR.TURN_ANGLE = turn_ang
    cfg.SIMULATOR.TILT_ANGLE = tilt_ang
    cfg.freeze()
    return habitat.sims.make_sim(cfg.SIMULATOR.TYPE, config=cfg.SIMULATOR)


def stamped_images(input_dir):
    img_path = os.path.join(input_dir, "x0.000_y0.000_z0.000_r0_rgb.jpeg")
    yield (0, 0, 0, 0), cv2.imread(img_path)
    for img_path in glob.iglob(os.path.join(input_dir, "*_rgb.jpeg")):
        filename = os.path.basename(img_path)
        x_real, y_real, z_real, a_real = (float(s[1:]) for s in filename.split('_')[:-1])
        a_real = numpy.radians(a_real)
        yield (x_real, y_real, z_real, a_real), cv2.imread(img_path)


def print_pos(label, x, y, z, a, same_line=False):
    a_deg = numpy.degrees(a)
    pre="\r" if same_line else "\n"
    print(f"{pre}{label: >12}: x={x:+7.3f} y={y:+7.3f} z={z:+7.3f} θ={a_deg:+4.0f}",
          end="" if same_line else "\n")


def step_sim(sim, key_code, sim_x, sim_y, sim_z, sim_a, sim_quat, step_size=STEP_SIZE):
    obs = None
    reject = False
    if key_code == ord('w'):
        obs = sim.step(1)
    elif key_code == ord('q'):
        obs = sim.step(2)
    elif key_code == ord('e'):
        obs = sim.step(3)
    elif key_code == ord('i'):
        obs = sim.step(4)
    elif key_code == ord('k'):
        obs = sim.step(5)
    elif key_code == ord('r'):
        obs = sim.reset()
    elif key_code == ord('s'): # BACKWARD
        x = sim_x + step_size * numpy.sin(sim_a)
        z = sim_z + step_size * numpy.cos(sim_a)
        obs = sim.get_observations_at(numpy.array([x, sim_y, z]), sim_quat, True)
    elif key_code == ord('a'): # STRAFE_LEFT
        x = sim_x - step_size * numpy.cos(sim_a)
        z = sim_z + step_size * numpy.sin(sim_a)
        obs = sim.get_observations_at(numpy.array([x, sim_y, z]), sim_quat, True)
    elif key_code == ord('d'): # STRAFE_RIGHT
        x = sim_x + step_size * numpy.cos(sim_a)
        z = sim_z - step_size * numpy.sin(sim_a)
        obs = sim.get_observations_at(numpy.array([x, sim_y, z]), sim_quat, True)
    elif key_code == ord('x'):
        reject = True
    elif key_code == ord('c'):
        pass
    return obs, reject


def main(args):
    sim = setup_sim(args.step_size, args.turn_angle, args.tilt_angle)
    obs = sim.reset()

    print(USAGE)

    measure_cnt = 0
    offsets = []
    for (real_x, real_y, real_z, real_a), real_obs in stamped_images(args.input_dir):
        cv2.imshow("Real robot", real_obs)
        print_pos("Real robot:", real_x, real_y, real_z, real_a)
        while True:
            state = sim.get_agent_state()
            sim_x, sim_y, sim_z = state.position
            alpha, sim_a, gamma = quaternion.as_euler_angles(state.rotation)
            if abs(alpha) > 1.5 and abs(gamma) > 1.5:
                sim_a *= -1
            print_pos("Simulation", sim_x, sim_y, sim_z, sim_a, True)

            cv2.imshow("Simulation", obs["rgb"][:, :, ::-1])
            key_code = cv2.waitKey()
            nxt_obs, reject = step_sim(sim, key_code, sim_x, sim_y, sim_z, sim_a,
                                       state.rotation, args.step_size)
            if nxt_obs is None:
                break
            else:
                obs = nxt_obs
        if reject:
            continue
        off_a = sim_a - real_a
        off_x = sim_x + real_x * numpy.sin(off_a) + real_y * numpy.cos(off_a)
        off_y = sim_y
        off_z = sim_z + real_x * numpy.cos(off_a) - real_y * numpy.sin(off_a)
        offsets.append([off_x, off_y, off_z, off_a])
        print_pos("Offset", off_x, off_y, off_z, off_a)

        measure_cnt += 1
        if measure_cnt == args.n_measures:
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main(parse_args())
