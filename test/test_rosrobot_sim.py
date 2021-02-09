from habitat_sim2real import ROSRobot, get_config

cfg = get_config("configs/locobot_pointnav_real.yaml")
sim = ROSRobot(cfg.SIMULATOR)

obs = sim.reset()
print(obs.keys())
