#!/usr/bin/env python
import os
os.environ["GLOG_minloglevel"] = "2"
os.environ["MAGNUM_LOG"] = "quiet"

import logging
import cv2
import numpy
import quaternion
import habitat
habitat.logger.setLevel(logging.ERROR)

HFOVS = (60, 70)
RATIOS = (1 / 1, 4 / 3,)
POSITION = numpy.array([7.336, 0.047, 0.860])
ANGLE = numpy.radians(161)
ROTATION = quaternion.quaternion(numpy.cos(0.5 * ANGLE), 0, numpy.sin(0.5 * ANGLE), 0)

cfg = habitat.get_config("configs/locobot_pointnav_citi_sim.yaml").SIMULATOR

for hfov in HFOVS:
    for ratio in RATIOS:
        cfg.defrost()
        cfg.AGENT_0.SENSORS = ["RGB_SENSOR"]
        cfg.RGB_SENSOR.WIDTH = int(ratio * cfg.RGB_SENSOR.HEIGHT)
        cfg.RGB_SENSOR.HFOV = hfov
        cfg.freeze()

        print("Rendering observations at x={0:.3f}, y={1:.3f}, a={3:.3f}".format(*POSITION, ANGLE))
        print("using aspect ratio={:.2f} and horizontal FoV={:.0f}.".format(ratio, hfov))

        e = 0.5 * cfg.RGB_SENSOR.WIDTH * numpy.tan(0.5 * numpy.radians(hfov))
        vfov = 2 * numpy.degrees(numpy.arctan(0.5 * cfg.RGB_SENSOR.HEIGHT / e))
        print("WxH={:.0f}x{:.0f}, vertical FoV={:.0f}".format(cfg.RGB_SENSOR.WIDTH, cfg.RGB_SENSOR.HEIGHT, vfov))

        sim = habitat.sims.make_sim(cfg.TYPE, config=cfg)
        obs = sim.get_observations_at(POSITION, ROTATION)
        sim.close()

        cv2.imshow("fov={:.0f} r={:.2f}".format(hfov, ratio), obs["rgb"][:, :, ::-1])

real = cv2.imread("out/real/expe_man_ctrl/x0.000_y0.000_z0.000_r0_rgb.jpeg")
cv2.imshow("Real", real)
cv2.waitKey()
