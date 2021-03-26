import os
os.environ["GLOG_minloglevel"] = "2"
os.environ["MAGNUM_LOG"] = "quiet"
import logging

import argparse
import numpy as np
from scipy.spatial import KDTree
import tqdm
import gzip

import habitat
habitat.logger.setLevel(logging.ERROR)
from habitat.tasks.nav.nav import NavigationGoal, NavigationEpisode


DEFAULT_ARGS = {"config_path": "configs/locobot_multigoal_pointnav_citi_sim.yaml",
                "n_episodes": 1000,
                "episodes_len": 10,
                "min_dist": 1.0,
                "max_dist": 7.0,
                "min_dist_ratio": 1.05,
                "if_exist": "exit"}
N_POINTS = 6000
MAX_RETRIES = 10
MIN_ISLAND_RADIUS = 1.5
EPS = 1e-5

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-path", "-c")
    parser.add_argument("--n-episodes", "-n", type=int)
    parser.add_argument("--episodes-len", "-l", type=int)
    parser.add_argument("--min-dist", type=float)
    parser.add_argument("--max-dist", type=float)
    parser.add_argument("--min-dist-ratio", type=float)
    parser.add_argument("--seed", "-s", type=int)
    parser.add_argument("--if-exist", choices=["override", "append", "exit"])
    parser.add_argument("extra_cfg", nargs=argparse.REMAINDER)

    parser.set_defaults(**DEFAULT_ARGS)
    return parser.parse_args()


def sample_point(sim, height=None, radius=MIN_ISLAND_RADIUS):
    pt = sim.sample_navigable_point()
    while sim.island_radius(pt) < radius or (height is not None and abs(pt[1] - height) > EPS):
        pt = sim.sample_navigable_point()
    return pt


def allow_straight(rng, d, euc_d, min_dist_ratio):
    r = (d + EPS) / euc_d
    return r >= min_dist_ratio or rng.random() < 20 * (r - 0.98)**2


def make_episode(rng, ep_id, scene_id, src, destinations, success_dist, tot_geo_dist):
    a = 2 * np.pi * rng.random()
    q = [0, np.sin(a / 2), 0, np.cos(a / 2)]
    goals = [NavigationGoal(position=dst, radius=success_dist) for dst in destinations]
    return NavigationEpisode(episode_id=ep_id, scene_id=scene_id,
                             start_position=src, start_rotation=q, goals=goals,
                             info={"geodesic_distance": tot_geo_dist})


def main(args):
    cfg = habitat.get_config(args.config_path, args.extra_cfg)
    out_fpath = cfg.DATASET.DATA_PATH.format(split=cfg.DATASET.SPLIT)
    os.makedirs(os.path.dirname(out_fpath), exist_ok=True)

    rng = np.random.default_rng(args.seed)

    try:
        dataset = habitat.make_dataset(cfg.DATASET.TYPE, config=cfg.DATASET)
        if args.if_exist == "exit":
            print("Dataset already exists! " \
                    + "Change the value of --if-exist to 'append' or 'override'")
            return
        elif args.if_exist == "override":
            dataset.episodes = []
    except FileNotFoundError:
        dataset = habitat.make_dataset(cfg.DATASET.TYPE)
    sim = habitat.sims.make_sim(cfg.SIMULATOR.TYPE, config=cfg.SIMULATOR)
    sim.seed(rng.integers(1, 100000))
    height = sim.get_agent_state().position[1]

    points = np.array([sample_point(sim, height) for _ in range(N_POINTS)])
    tree = KDTree(points)

    init_len = len(dataset.episodes)
    ep_cnt = init_len
    with tqdm.tqdm(total=args.n_episodes) as progress:
        while len(dataset.episodes) < init_len + args.n_episodes:
            src = rng.choice(points)
            destinations = []
            prv = src
            tot_d = 0
            for _ in range(args.episodes_len):
                too_near_indices = tree.query_ball_point(prv, 0.8 * args.min_dist)
                near_indices = tree.query_ball_point(prv, args.max_dist)
                indices = list(set(near_indices) - set(too_near_indices))
                for _ in range(MAX_RETRIES):
                    nxt = rng.choice(points[indices])
                    euc_d = np.sqrt(np.sum((nxt - prv)**2))
                    d = sim.geodesic_distance(prv, nxt)
                    if np.isfinite(d) and args.min_dist <= d <= args.max_dist \
                            and allow_straight(rng, d, euc_d, args.min_dist_ratio):
                                break
                else:
                    break
                tot_d += d
                destinations.append(nxt)
                prv = nxt
            else:
                episode = make_episode(rng, str(ep_cnt), cfg.SIMULATOR.SCENE, src, destinations,
                                       cfg.TASK.SUCCESS.SUCCESS_DISTANCE, tot_d)
                dataset.episodes.append(episode)
                ep_cnt += 1
                progress.update()
    with gzip.open(out_fpath, "wt") as f:
        f.write(dataset.to_json())


if __name__ == "__main__":
    main(parse_args())
