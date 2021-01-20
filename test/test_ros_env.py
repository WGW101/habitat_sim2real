#!/usr/bin/env python3

import habitat_sim2real.envs.ros_env
import habitat_sim2real.sims.ros.rosrobot_sim

import habitat

from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.config.default import get_config

cfg = get_config("configs/locobot_ppo_real.yaml")
print(cfg.ENV_NAME)
print(cfg.TASK_CONFIG.SIMULATOR.TYPE)

env_cls = baseline_registry.get_env(cfg.ENV_NAME)
env = env_cls(cfg)
print(env)

print(env._env._sim)

env.reset()
print(env._env.current_episode)
