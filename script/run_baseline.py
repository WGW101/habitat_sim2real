#!/usr/bin/env python3

import os
os.environ["GLOG_minloglevel"] = "2"

import argparse

import habitat_sim2real

from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.config.default import get_config


CFG_PATH = "configs/locobot_ppo_citi_sim.yaml"


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

    trainer_cls = baseline_registry.get_trainer(cfg.TRAINER_NAME)
    trainer = trainer_cls(cfg)

    if args.run_type == "train":
        trainer.train()
    elif args.run_type == "eval":
        trainer.eval()


if __name__ == "__main__":
    main(parse_args())
