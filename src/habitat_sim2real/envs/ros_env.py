from habitat.core.env import Env
from habitat.tasks.nav.nav import NavigationGoal, NavigationEpisode

from habitat_baselines.common.environments import NavRLEnv
from habitat_baselines.common.baseline_registry import baseline_registry

from habitat_sim2real.sims.ros.default_cfg import merge_ros_config


class ROSEnv(Env):
    def __init__(self, config, dataset=None):
        config = merge_ros_config(config)
        super().__init__(config, None)
        self.episode_count = 0

    def reset(self):
        self._reset_stats()

        if self._current_episode:
            self._episodes.append(self._current_episode)
            self.episode_count += 1

        state = self._sim.get_agent_state()
        goal = NavigationGoal(position=self._sim.sample_navigable_point(),
                              radius=self._config.TASK.SUCCESS.SUCCESS_DISTANCE)
        self._current_episode = NavigationEpisode(episode_id=str(self.episode_count),
                                                  scene_id="REAL",
                                                  start_position=state.position.tolist(),
                                                  start_rotation=state.rotation.tolist(),
                                                  goals=[goal])

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

        self._env = ROSEnv(config.TASK_CONFIG, dataset)
        self.observation_space = self._env.observation_space
        self.action_space = self._env.action_space
        self.number_of_episodes = self._env.number_of_episodes
        self.reward_range = self.get_reward_range()
