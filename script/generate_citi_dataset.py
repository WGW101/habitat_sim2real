#!/usr/bin/env python3

import os
os.environ["GLOG_minloglevel"] = "2"
os.environ["MAGNUM_LOG"] = "quiet"

import argparse
import random
import gzip
import tqdm
import math
from scipy.spatial.distance import pdist as pairwise_distance

import habitat
from habitat.tasks.nav.nav import NavigationGoal, NavigationEpisode


CFG_PATH = "configs/locobot_pointnav_citi_sim_simple.yaml"
N_EPISODES = 300
DIFFICULTIES = ("very easy", "easy", "medium", "hard")
DIFFICULTY_BOUNDS = (1.0, 3.0, 7.0, 13.0, 20.0)
DIFFICULTY_RATIOS = "50, 15, 20, 15"
MIN_DIST_RATIO = 1.1
MIN_ISLAND_RADIUS = 1.5


def parse_args():
    parser = argparse.ArgumentParser(description="Generate random pointnav train/test datasets")
    parser.add_argument("--config-path", "-c", default=CFG_PATH, help="Path to config file")
    parser.add_argument("--n-episodes", "-n", type=int, default=N_EPISODES,
                        help="Number of episodes to include in the train dataset")
    parser.add_argument("--difficulty-ratios", "-r", default=DIFFICULTY_RATIOS,
                        help="Relative number of 'very easy', 'easy'," \
                             + " 'medium' and 'hard' episodes")
    parser.add_argument("--seed", "-s", type=int, default=random.randint(10000,100000),
                        help="Seed used to initialize the RNG")
    parser.add_argument("extra_cfg", nargs=argparse.REMAINDER,
                        help="Extra config options as 'KEY value' pairs")
    return parser.parse_args()


def parse_difficulty_ratios(args):
    ratios = [float(r) for r in args.difficulty_ratios.split(',')]
    if any(r < 0 for r in ratios) or len(ratios) != 4:
        raise ValueError(f"Invalid difficulty ratios argument: '{args.difficulty_ratios}'")
    tot = sum(ratios)
    return {k: (int(r * args.n_episodes / tot), min_d, max_d)
            for k, r, min_d, max_d in zip(DIFFICULTIES, ratios,
                                          DIFFICULTY_BOUNDS[:-1], DIFFICULTY_BOUNDS[1:])}


def sample_point(sim, radius=MIN_ISLAND_RADIUS):
    pt = sim.sample_navigable_point()
    while sim.island_radius(pt) < radius:
        pt = sim.sample_navigable_point()
    return pt


def main(args):
    cfg = habitat.get_config(args.config_path, args.extra_cfg)
    out_file = cfg.DATASET.DATA_PATH.format(split=cfg.DATASET.SPLIT)
    os.makedirs(os.path.dirname(out_file), exist_ok=True)

    difficulties = parse_difficulty_ratios(args)

    sim = habitat.sims.make_sim(cfg.SIMULATOR.TYPE, config=cfg.SIMULATOR)
    sim.seed(args.seed)
    dataset = habitat.datasets.make_dataset(cfg.DATASET.TYPE)

    n_pts = 2 * args.n_episodes
    nav_pts = [sample_point(sim) for _ in range(n_pts)]
    pairs = [(i, j) for i in range(n_pts - 1) for j in range(i, n_pts)]

    euc_dists = pairwise_distance(nav_pts)
    tot_cnt = 0
    with tqdm.tqdm(total=args.n_episodes) as progress:
        for k, (n_ep, min_d, max_d) in difficulties.items():
            candidates = [(p, d) for p, d in zip(pairs, euc_dists)
                          if 0.8 * min_d <= d <= 1.2 * max_d]
            cnt = 0
            for (i, j), euc_d in candidates:
                src = nav_pts[i]
                dst = nav_pts[j]
                d = sim.geodesic_distance(src, dst)
                if min_d <= d <= max_d and d / euc_d >= MIN_DIST_RATIO:
                    a = 2 * math.pi * random.random()
                    q = [0, math.sin(a / 2), 0, math.cos(a / 2)]
                    goal = NavigationGoal(position=dst,
                                          radius=cfg.TASK.SUCCESS.SUCCESS_DISTANCE)
                    episode = NavigationEpisode(episode_id=str(tot_cnt),
                                                scene_id=cfg.SIMULATOR.SCENE,
                                                start_position=src,
                                                start_rotation=q,
                                                goals=[goal],
                                                info={"difficulty": k,
                                                      "geodesic_distance": d})
                    dataset.episodes.append(episode)
                    cnt += 1
                    tot_cnt += 1
                    progress.update(1)
                    if cnt == n_ep:
                        break

    with gzip.open(out_file, "wt") as f:
        f.write(dataset.to_json())


if __name__ == "__main__":
    main(parse_args())
