import habitat
import habitat_sim2real

cfg = habitat.get_config("configs/locobot_pointnav_citi_sim.yaml").SIMULATOR
cfg.defrost()
cfg.DEPTH_SENSOR.TYPE = "RealisticHabitatSimDepthSensor"
cfg.freeze()

viewer = habitat_sim2real.BaseSimulatorViewer(cfg)
viewer.run()
