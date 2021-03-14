#!/usr/bin/env python

import os
os.environ["GLOG_minloglevel"] = "2"
os.environ["MAGNUM_LOG"] = "quiet"
import logging

import glob
import random
import argparse

import habitat
habitat.logger.setLevel(logging.ERROR)
import cv2
import numpy
import quaternion


IN_DIR = "out/traj_cap/real"
OUT_DIR = "out/traj_cap/sim"
CFG_PATH = "configs/locobot_pointnav_citi_sim.yaml"
STEP_SIZE = 0.05
TURN_ANGLE = 1
TILT_ANGLE = 1
INIT_OFF_X = 7.336
INIT_OFF_Z = 0.260
INIT_OFF_A = 163

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
    parser.add_argument("--output-dir", "-o", default=OUT_DIR)
    parser.add_argument("--cfg-path", "-c", default=CFG_PATH)
    parser.add_argument("--step-size", type=float, default=STEP_SIZE)
    parser.add_argument("--turn-angle", type=float, default=TURN_ANGLE)
    parser.add_argument("--tilt-angle", type=float, default=TILT_ANGLE)
    parser.add_argument("--no-grid", action="store_true")
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
    with open(os.path.join(input_dir, "chrono_list.txt")) as f:
        for l in f:
            filepath = l.strip()
            filename = os.path.basename(filepath)
            real_x, real_y, real_z, real_a = (float(s[1:]) for s in filename.split('_')[:-1])
            real_a = numpy.radians(real_a)
            yield [real_x, real_y, real_z], real_a, cv2.imread(filepath), filepath


def draw_grid(img):
    disp = img.copy()
    H, W, _ = img.shape

    for i in range(1, 12, 2):
        cv2.line(disp, (i * W // 12, 0), (i * W // 12, H), (0, 255, 0))
        cv2.line(disp, (0, i * H // 12), (W, i * H // 12), (0, 255, 0))

    for i in (1, 2, 4, 5):
        cv2.line(disp, (i * W // 6, 0), (i * W // 6, H), (255, 0, 0))
        cv2.line(disp, (0, i * H // 6), (W, i * H // 6), (255, 0, 0))

    cv2.line(disp, (W // 2, 0), (W // 2, H), (0, 0, 255))
    cv2.line(disp, (0, H // 2), (W, H // 2), (0, 0, 255))

    return disp


def print_pos(label, pos, a, same_line=False):
    a_deg = numpy.degrees(a)
    pre="\r" if same_line else "\n"
    print("{}{: >12}: x={:+7.3f} y={:+7.3f} z={:+7.3f} θ={:+4.0f}".format(
        pre, label, *pos, a_deg), end="" if same_line else "\n")


def step_sim(sim, key_code, sim_pos, sim_a, sim_quat, step_size=STEP_SIZE):
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
        x = sim_pos[0] + step_size * numpy.sin(sim_a)
        z = sim_pos[2] + step_size * numpy.cos(sim_a)
        obs = sim.get_observations_at(numpy.array([x, sim_pos[1], z]), sim_quat, True)
    elif key_code == ord('a'): # STRAFE_LEFT
        x = sim_pos[0] - step_size * numpy.cos(sim_a)
        z = sim_pos[2] + step_size * numpy.sin(sim_a)
        obs = sim.get_observations_at(numpy.array([x, sim_pos[1], z]), sim_quat, True)
    elif key_code == ord('d'): # STRAFE_RIGHT
        x = sim_pos[0] + step_size * numpy.cos(sim_a)
        z = sim_pos[2] - step_size * numpy.sin(sim_a)
        obs = sim.get_observations_at(numpy.array([x, sim_pos[1], z]), sim_quat, True)
    elif key_code == ord('x'):
        reject = True
    elif key_code == ord('c'):
        pass
    return obs, reject


def save_obs(output_dir, sim_obs, sim_pos, sim_a):
    a_deg = numpy.degrees(sim_a)
    filename = "x{:.3f}_y{:.3f}_z{:.3f}_r{:.0f}_rgb.jpeg".format(*sim_pos, a_deg)
    filepath = os.path.join(output_dir, filename)
    cv2.imwrite(filepath, sim_obs["rgb"][:, :, ::-1])
    depth = (sim_obs["depth"] * 255).astype(numpy.uint8)
    cv2.imwrite(os.path.join(output_dir, filename.replace("rgb", "depth")), depth)
    return filepath


def calc_offset(sim_pos, sim_a, real_pos, real_a):
    a = sim_a - real_a
    if a > numpy.pi:
        a -= 2*numpy.pi
    if a < -numpy.pi:
        a += 2*numpy.pi
    x = sim_pos[0] + real_pos[0] * numpy.sin(a) + real_pos[1] * numpy.cos(a)
    y = sim_pos[1]
    z = sim_pos[2] + real_pos[0] * numpy.cos(a) - real_pos[1] * numpy.sin(a)
    return [x, y, z], a


def apply_offset(sim, off_pos, off_a, real_pos, real_a):
    x = off_pos[0] - real_pos[0] * numpy.sin(off_a) - real_pos[1] * numpy.cos(off_a)
    y = off_pos[1]
    z = off_pos[2] - real_pos[0] * numpy.cos(off_a) + real_pos[1] * numpy.sin(off_a)

    a = off_a + real_a
    if a > numpy.pi:
        a -= 2*numpy.pi
    if a < -numpy.pi:
        a += 2*numpy.pi

    pos = numpy.array([x, y, z])
    rot = quaternion.quaternion(numpy.cos(0.5 * a), 0, numpy.sin(0.5 * a), 0)
    return sim.get_observations_at(pos, rot, True)


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    match_file = open(os.path.join(args.output_dir, "match.csv"), 'w')
    match_file.write("# REAL RGB IMG PATH,    # SIM RGB IMG PATH\n")

    sim = setup_sim(args.step_size, args.turn_angle, args.tilt_angle)
    off_pos = [INIT_OFF_X, 0, INIT_OFF_Z]
    off_a = numpy.radians(INIT_OFF_A)
    print(USAGE)

    measure_cnt = 0
    for real_pos, real_a, real_obs, real_filepath in stamped_images(args.input_dir):
        cv2.imshow("Real robot", real_obs if args.no_grid else draw_grid(real_obs))
        print_pos("Real robot", real_pos, real_a)

        sim_obs = apply_offset(sim, off_pos, off_a, real_pos, real_a)

        while True:
            state = sim.get_agent_state()
            alpha, sim_a, gamma = quaternion.as_euler_angles(state.rotation)
            if abs(alpha) > 1.5 and abs(gamma) > 1.5:
                sim_a *= -1
            print_pos("Simulation", state.position, sim_a, True)
            rgb = sim_obs["rgb"][:, :, ::-1]
            cv2.imshow("Simulation", rgb if args.no_grid else draw_grid(rgb))
            key_code = cv2.waitKey()
            nxt_obs, reject = step_sim(sim, key_code, state.position, sim_a,
                                       state.rotation, args.step_size)
            if nxt_obs is None:
                break
            else:
                sim_obs = nxt_obs

        if reject:
            continue

        off_pos, off_a = calc_offset(state.position, sim_a, real_pos, real_a)
        print_pos("Offset", off_pos, off_a)

        sim_filepath = save_obs(args.output_dir, sim_obs, state.position, sim_a)
        match_file.write(real_filepath + ",    " + sim_filepath + "\n")

    cv2.destroyAllWindows()
    match_file.close()


if __name__ == "__main__":
    main(parse_args())
