import sys
import os
os.environ["GLOG_minloglevel"] = "2"
os.environ["MAGNUM_LOG"] = "quiet"
import logging
import math

import tqdm
import numpy as np
import cv2

import habitat


CFG_PATH = "/home/wgw/py_ws/wgw101/habitat_sim2real/configs/locobot_pointnav_citi_sim.yaml"
EXPL_RADIUS = 3 # m
RESOLUTION = 0.02 # m / px


habitat.logger.setLevel(logging.ERROR)
cfg = habitat.get_config(CFG_PATH, sys.argv[1:])

data = habitat.make_dataset(cfg.DATASET.TYPE, config=cfg.DATASET)

with habitat.sims.make_sim(cfg.SIMULATOR.TYPE, config=cfg.SIMULATOR) as sim:
    height = sim.get_agent_state().position[1]
    nav_mask = sim.pathfinder.get_topdown_view(RESOLUTION, height)
    origin, _ = sim.pathfinder.get_bounds()
    origin[1] = height
j_i = np.stack([(np.array(episode.start_position) - origin) / RESOLUTION
                for episode in data.episodes], 0).astype(np.int64)

starts_mask = np.zeros(nav_mask.shape, dtype=np.float32)
for j, _, i in j_i:
    starts_mask[i, j] += 1.0

cv2.imshow("Test", starts_mask)
cv2.waitKey()

sigma = EXPL_RADIUS / (3 * RESOLUTION)
kern = cv2.getGaussianKernel(2 * int(5 * sigma) + 1, sigma)
starts_img = cv2.sepFilter2D(starts_mask, -1, kern, kern)

starts_img = starts_img / (np.mean(starts_img) + 3 * np.std(starts_img))
cv2.imshow("Test", starts_img)
cv2.waitKey()

starts_img = (255 * starts_img.clip(None, 1.0)).astype(np.uint8)
disp = np.full(nav_mask.shape + (3,), 127, dtype=np.uint8)
disp[nav_mask] = cv2.applyColorMap(starts_img, cv2.COLORMAP_JET)[nav_mask]

edges = np.zeros_like(nav_mask)
edges[:-1, :-1] = (nav_mask[:-1, :-1] != nav_mask[1:, :-1]) \
                | (nav_mask[:-1, :-1] != nav_mask[:-1, 1:])
disp[edges] = 0

cv2.imshow("Heatmap", disp)
cv2.waitKey()
