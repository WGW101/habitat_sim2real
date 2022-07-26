import habitat
from habitat_sim2real import BaseSimulatorViewer

cfg = habitat.get_config("configs/locobot_pointnav_citi_sim.yaml")
cfg.defrost()
cfg.SIMULATOR.SCENE = "data/scene_datasets/nle/chateau.glb"
cfg.SIMULATOR.RGB_SENSOR.WIDTH = 640
cfg.SIMULATOR.RGB_SENSOR.HEIGHT = 480
cfg.SIMULATOR.DEPTH_SENSOR.WIDTH = 640
cfg.SIMULATOR.DEPTH_SENSOR.HEIGHT = 480
cfg.freeze()
viewer = BaseSimulatorViewer(cfg.SIMULATOR, draw_origin=True)
viewer.run()
