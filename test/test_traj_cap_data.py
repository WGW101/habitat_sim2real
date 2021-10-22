import os
os.environ["GLOG_minloglevel"] = "2"
os.environ["MAGNUM_LOG"] = "quiet"
import logging

import numpy as np
import quaternion
import cv2

import habitat
habitat.logger.setLevel(logging.ERROR)
from habitat.utils.visualizations.maps import get_topdown_map_from_sim, colorize_topdown_map


cfg = habitat.get_config("configs/locobot_pointnav_citi_sim.yaml").SIMULATOR
sim = habitat.sims.make_sim(cfg.TYPE, config=cfg)

topdown = colorize_topdown_map(get_topdown_map_from_sim(sim))
b = sim.pathfinder.get_bounds()
orig = np.array([b[0][0], b[0][2]])
res = np.array([(b[1][0] - b[0][0]) / topdown.shape[1],
                   (b[1][2] - b[0][2]) / topdown.shape[0]])

with open("out/traj_cap/21-01-26/sim/match.csv") as f:
    next(f) # skip header
    paths = [[item.strip() for item in l.split(',')] for l in f]
real_paths, sim_paths = zip(*paths)

sim_poses = [[float(s[1:]) for s in path.split('/')[-1].split('_')[:4]] for path in sim_paths]
sim_poses = np.array(sim_poses)
pos = sim_poses[:, :3]
a = np.radians(sim_poses[:, 3])
rot = np.zeros((a.shape[0], 4))
rot[:, 0] = np.cos(0.5 * a)
rot[:, 2] = np.sin(0.5 * a)
rot = quaternion.from_float_array(rot)

map_pos = ((pos[:, ::2] - orig) / res).astype(np.int64)
fwd = quaternion.quaternion()
fwd.vec = sim.forward_vector
heads = quaternion.as_float_array(rot * fwd * rot.conjugate())[:, 1::2]
head_ends = (map_pos + 10 * heads).astype(np.int64)

cv2.namedWindow("Map")
selected = None
update = True
def on_mouse(event, x, y, flags, param):
    global selected, update
    if event == cv2.EVENT_LBUTTONDOWN:
        selected, _ = min(enumerate(map_pos),
                          key=lambda iuv: (iuv[1][0] - x)**2 + (iuv[1][1] - y)**2)
        update = True
cv2.setMouseCallback("Map", on_mouse)

while True:
    if update:
        disp = topdown.copy()
        for i, (s, e) in enumerate(zip(map_pos, head_ends)):
            if i == selected:
                continue
            cv2.circle(disp, tuple(s), 5, (255, 0, 0), -1)
            cv2.line(disp, tuple(s), tuple(e), (255, 0, 0), 3)
        if selected is not None:
            s = map_pos[selected]
            e = head_ends[selected]
            cv2.circle(disp, tuple(s), 5, (0, 255, 0), -1)
            cv2.line(disp, tuple(s), tuple(e), (0, 255, 0), 3)

            real_rgb = cv2.imread(real_paths[selected])
            sim_rgb = cv2.imread(sim_paths[selected])
            cv2.imshow("Obs", np.vstack((real_rgb, sim_rgb)))
        cv2.imshow("Map", disp)
        update = False
    c = cv2.waitKey(30)
    if c == ord('x'):
        break
    elif c == ord('r'):
        selected = None
        update = True
