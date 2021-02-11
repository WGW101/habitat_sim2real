from habitat_sim2real import ROSRobot, get_config

cfg = get_config("configs/locobot_pointnav_real.yaml")
sim = ROSRobot(cfg.SIMULATOR)

obs = sim.reset()
print(obs.keys())

while True:
    pos = sim.get_agent_state().position.tolist()
    print("Pos.:", pos)
    goal_pos = sim.sample_navigable_point()
    print("Goal:", goal_pos)
    sim.publish_episode_goal(goal_pos)
    d = sim.geodesic_distance(pos, goal_pos)
    print("Distance:", d)
    if input():
        break
