import os
os.environ["GLOG_minloglevel"] = "2"
os.environ["MAGNUM_LOG"] = "quiet"

import argparse
import math
import tqdm
from itertools import groupby
from matplotlib import pyplot

import habitat
from habitat.datasets.utils import get_action_shortest_path


CFG_PATH = "configs/locobot_pointnav_citi_sim.yaml"
MAX_EPISODE_STEPS = 500


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze difficulty of pointnav datasets")
    parser.add_argument("--config-path", "-c", default=CFG_PATH, help="Path to config file")
    parser.add_argument("--gen-shortest-path", "-g", action="store_true",
                        help="Generate shortest path to get min #steps")
    parser.add_argument("--use-densities", "-d", action="store_true",
                        help="Plot using densities instead of #samples")
    parser.add_argument("extra_cfg", nargs=argparse.REMAINDER,
                        help="Extra config options as 'KEY value' pairs")
    return parser.parse_args()


def scene_id(ep):
    return ep.scene_id

def difficulty(ep):
    return ep.info.get("difficulty", "unknown")

def difficulty_order(ep):
    order = {k:v for v, k in enumerate(("very easy", "easy", "medium", "hard"))}
    info = difficulty(ep)
    return (order.get(info, 99), info)


def main(args):
    cfg = habitat.get_config(args.config_path, args.extra_cfg)
    dataset = habitat.make_dataset(cfg.DATASET.TYPE, config=cfg.DATASET)

    for episode in dataset.episodes:
        d = math.sqrt(sum((xs- xg)**2 for xs, xg in zip(episode.start_position,
                                                        episode.goals[0].position)))
        episode.info["euclidean_distance"] = d

    if args.gen_shortest_path:
        with tqdm.tqdm(total=dataset.num_episodes) as pbar:
            for scene, episodes in groupby(sorted(dataset.episodes, key=scene_id),
                                           key=scene_id):
                cfg.defrost()
                cfg.SIMULATOR.SCENE = scene
                cfg.freeze()
                sim = habitat.sims.make_sim(cfg.SIMULATOR.TYPE, config=cfg.SIMULATOR)
                for episode in episodes:
                    path = get_action_shortest_path(sim,
                                                    episode.start_position,
                                                    episode.start_rotation,
                                                    episode.goals[0].position,
                                                    episode.goals[0].radius,
                                                    MAX_EPISODE_STEPS)
                    if path and len(path) < MAX_EPISODE_STEPS:
                        episode.info["min_num_steps"] = len(path)
                    pbar.update(1)

    counts = {}
    geo_dists = []
    num_steps = []
    dist_ratios = []
    for k, episodes in groupby(sorted(dataset.episodes, key=difficulty_order), key=difficulty):
        episodes = list(episodes)
        counts[k] = len(episodes)
        geo_dists.append([ep.info["geodesic_distance"] for ep in episodes])
        if args.gen_shortest_path:
            num_steps.append([ep.info["min_num_steps"] for ep in episodes])
        dist_ratios.append([ep.info["geodesic_distance"] / ep.info["euclidean_distance"]
                            for ep in episodes])

    cmap = pyplot.get_cmap("jet")
    colors = [cmap(i / len(counts)) for i, _ in enumerate(counts)]
    fig, axes = pyplot.subplots(4 if args.gen_shortest_path else 3, 1)
    for ax in axes:
        ax.set_ylabel("Density" if args.use_densities else "#samples")
    axes[0].set_xlabel("Difficulty")
    axes[0].bar(*zip(*counts.items()), color=colors)
    axes[1].set_xlabel("Geodesic distance")
    axes[1].hist(geo_dists, stacked=True, density=args.use_densities, bins=20, color=colors)
    if args.gen_shortest_path:
        axes[2].set_xlabel("#steps in shortest path")
        axes[2].hist(num_steps, stacked=True, density=args.use_densities, bins=20, color=colors)
    axes[-1].set_xlabel("Geodesic/Euclidian distance ratio")
    axes[-1].hist(dist_ratios, stacked=True, density=args.use_densities, bins=20, color=colors)

    fig.suptitle("_".join(cfg.DATASET.DATA_PATH.split("/")[2:5]))
    fig.set_tight_layout(True)
    pyplot.show()


if __name__ == "__main__":
    main(parse_args())
