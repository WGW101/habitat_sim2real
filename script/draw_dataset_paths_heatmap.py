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
RESOLUTION = 0.02


habitat.logger.setLevel(logging.ERROR)
cfg = habitat.get_config(CFG_PATH, sys.argv[1:])

data = habitat.make_dataset(cfg.DATASET.TYPE, config=cfg.DATASET)

with habitat.sims.make_sim(cfg.SIMULATOR.TYPE, config=cfg.SIMULATOR) as sim:
    height = sim.get_agent_state().position[1]
    nav_mask = sim.pathfinder.get_topdown_view(RESOLUTION, height)
    origin, _ = sim.pathfinder.get_bounds()
    origin[1] = height
    paths_xyz = [np.array(sim.get_straight_shortest_path_points(episode.start_position,
                                                                episode.goals[0].position))
                 for episode in tqdm.tqdm(data.episodes, desc="Compute shortest paths")]

paths_img = np.zeros(nav_mask.shape, dtype=np.float32)
for xyz in tqdm.tqdm(paths_xyz, desc="Draw shortest paths"):
    ji = ((xyz - origin) / RESOLUTION)[:, [0, 2]].astype(np.int64)
    img = np.zeros_like(paths_img)
    cv2.polylines(img, [ji], False, 1.0, 3)
    paths_img += img

paths_img = paths_img / (np.mean(paths_img) + 3 * np.std(paths_img))
cv2.imshow("Test", paths_img)
cv2.waitKey()

paths_img = (255 * paths_img.clip(None, 1.0)).astype(np.uint8)
disp = np.full(nav_mask.shape + (3,), 127, dtype=np.uint8)
disp[nav_mask] = cv2.applyColorMap(paths_img, cv2.COLORMAP_JET)[nav_mask]

edges = np.zeros_like(nav_mask)
edges[:-1, :-1] = (nav_mask[:-1, :-1] != nav_mask[1:, :-1]) \
                | (nav_mask[:-1, :-1] != nav_mask[:-1, 1:])
disp[edges] = 0

cv2.imshow("Heatmap", disp)
cv2.waitKey()
