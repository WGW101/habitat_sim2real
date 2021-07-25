import os
import argparse
import gzip
import json

import tqdm


DEFAULT_ARGS = {"input_path": "data/datasets/pointnav/citi/v2/train/train.json.gz"}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-path", "-i")
    parser.add_argument("--output-path", "-o")
    parser.add_argument("extra_cfg", nargs=argparse.REMAINDER)
    parser.set_defaults(**DEFAULT_ARGS)
    return parser.parse_args()


def reverse_raw_episode(episode):
    episode["episode_id"] = "reversed_" + episode["episode_id"]
    start = episode["start_position"]
    goal = episode["goals"][0]["position"]
    episode["start_position"] = goal
    episode["goals"][0]["position"] = start
    return episode


def main(args):
    if args.output_path is None:
        prefix, *suffix = args.input_path.rsplit('/', 2)
        args.output_path = '/'.join((prefix + "_reversed", *suffix))

    print(f"Opening {args.input_path}")
    with gzip.open(args.input_path) as f:
        data = json.load(f)

    reversed_episodes = [reverse_raw_episode(episode)
                         for episode in tqdm.tqdm(data["episodes"], desc="Reversing episodes")]

    print(f"Saving {args.output_path}")
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with gzip.open(args.output_path, 'wt') as f:
        json.dump({"episodes": reversed_episodes}, f)
    print("Done")

if __name__ == "__main__":
    main(parse_args())
