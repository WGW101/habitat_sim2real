import cv2
import habitat
from habitat.utils.visualizations.maps import colorize_topdown_map

cfg = habitat.get_config("configs/locobot_pointnav_citi_sim.yaml")
cfg.defrost()
cfg.TASK.MEASUREMENTS.append("TOP_DOWN_MAP")
cfg.freeze()

dataset = habitat.make_dataset(cfg.DATASET.TYPE, config=cfg.DATASET)
env = habitat.Env(cfg, dataset)

obs = env.reset()
env.step(1) # top-down map not generated on reset...
map_info = env.get_metrics()["top_down_map"]
color_map = colorize_topdown_map(map_info["map"], map_info["fog_of_war_mask"])

cv2.imshow("Map", color_map)
cv2.waitKey()
