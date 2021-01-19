#!/usr/bin/env python3

import argparse

import envs.ros_env
import sims.ros.rosrobot_sim

from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.config.default import get_config


CONFIG_PATH = "config/locobot_citi_ppo.yaml"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-path", "-c", default=CFG_PATH, help="Path to config file")
    parser.add_argument("--run-type", "-r", choices=["train", "eval"], default="eval",
                        help="Toggle between training or evaluation")
    parser.add_argument("extra_cfg", nargs=argparse.REMAINDER,
                        help="Extra config options as 'KEY value' pairs")
    return parser.parse_args()


def main(args):
    cfg = get_config(args.config_path, args.extra_cfg)

    trainer_cls = baseline_registry.get_trainer(config.TRAINER_NAME)
    trainer = trainer_cls(config)

    if args.run_type == "train":
        trainer.train()
    elif args.run_type == "eval":
        train.eval()


if __name__ == "__main__":
    main(parse_args())
