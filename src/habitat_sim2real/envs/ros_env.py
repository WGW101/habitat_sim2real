import itertools
import os
import gzip
import time

from habitat.core.env import Env
from habitat.core.logging import logger
from habitat.tasks.nav.nav import NavigationGoal, NavigationEpisode
from habitat.datasets import make_dataset

from habitat_baselines.common.environments import NavRLEnv
from habitat_baselines.common.baseline_registry import baseline_registry


class ROSEnv(Env):
    def __init__(self, config, *args, **kwargs):
        super().__init__(config=config, *args, **kwargs)
        self._dataset = make_dataset(config.DATASET.TYPE)
        self._episodes = self._dataset.episodes
        self._current_episode = None
        self.t_str = time.strftime("%y-%m-%d_%H-%M-%S")

    def close(self):
        if not self._episodes \
            or self._episodes[-1].episode_id != self._current_episode.episode_id:
                self._episodes.append(self._current_episode)
        outpath = f"data/datasets/pointnav/real_online_demo/{self.t_str}/val/val.json.gz"
        os.makedirs(os.path.dirname(outpath))
        with gzip.open(outpath, 'wt') as f:
            f.write(self._dataset.to_json())
        super().close()

    def reset(self):
        self._reset_stats()

        if self._current_episode is not None:
            logger.info(f"{self._current_episode.episode_id}:"
                        + ", ".join(f"{k}={v:.2f}" for k, v in self.get_metrics().items()))
            self._episodes.append(self._current_episode)

        state = self._sim.get_agent_state()
        goal = NavigationGoal(position=self._sim.sample_navigable_point(),
                              radius=self._config.TASK.SUCCESS.SUCCESS_DISTANCE)
        self._current_episode = NavigationEpisode(
            episode_id=f"REAL_{self.t_str}_{len(self._episodes)}",
            scene_id="REAL",
            start_position=state.position.tolist(),
            start_rotation=[state.rotation.x, state.rotation.y,
                            state.rotation.z, state.rotation.w],
            goals=[goal]
        )

        observations = self._task.reset(episode=self._current_episode)
        self._task.measurements.reset_measures(episode=self._current_episode, task=self._task)
        return observations


@baseline_registry.register_env(name="ROSNavRLEnv")
class ROSNavRLEnv(NavRLEnv):
    def __init__(self, config, dataset=None):
        self._rl_config = config.RL
        self._core_env_config = config.TASK_CONFIG
        self._reward_measure_name = self._rl_config.REWARD_MEASURE
        self._success_measure_name = self._rl_config.SUCCESS_MEASURE

        self._previous_measure = None
        self._previous_action = None

        self._env = ROSEnv(config.TASK_CONFIG)
        self.observation_space = self._env.observation_space
        self.action_space = self._env.action_space
        self.number_of_episodes = self._env.number_of_episodes
        self.reward_range = self.get_reward_range()
