import sys
import numpy as np
import tqdm
import habitat

cfg = habitat.get_config("configs/locobot_pointnav_citi_sim.yaml", sys.argv[1:])
dataset = habitat.make_dataset(cfg.DATASET.TYPE, config=cfg.DATASET)
ep_iter = dataset.get_episode_iterator(cycle=False)
sim = habitat.sims.make_sim(cfg.SIMULATOR.TYPE, config=cfg.SIMULATOR)
init_pos = sim.get_agent_state().position

fail = []
success_cnt = 0
tot_cnt = 0

with tqdm.tqdm(ep_iter, total=dataset.num_episodes) as progress:
    for ep in progress:
        if ep.scene_id != sim._current_scene:
            cfg.SIMULATOR.defrost()
            cfg.SIMULATOR.SCENE = ep.scene_id
            cfg.SIMULATOR.freeze()
            sim.reconfigure(cfg.SIMULATOR)
        d = sim.geodesic_distance(init_pos, ep.start_position)
        if np.isfinite(d):
            success_cnt += 1
        else:
            fail.append(ep.episode_id)
        tot_cnt += 1
        progress.set_postfix_str("Success={:> 6.1%}".format(success_cnt / tot_cnt))
print("Success={:> 6.1%}".format(success_cnt / tot_cnt))
print("Episodes whose start could not be reached from init pos:")
print(fail)
