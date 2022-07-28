from typing import List, Dict, Any
import pickle
import numpy as np
from habitat.tasks.sequential_nav.sequential_nav import SequentialEpisode, SequentialSPL, SequentialNavigationTask, PPL, SequentialSuccess, Progress
from habitat.core.registry import registry

CACHED_DATA_ROOT = "cached_measurements/"


@registry.register_measure
class OnlineSequentialSPL(SequentialSPL):
    _cached_shortest_dist: Dict[str, float]
    _last_delta: float

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        if self._sim.habitat_config.TYPE == "ROS-Robot-v0":
            with open(CACHED_DATA_ROOT + "online_seq_spl.pkl", 'rb') as pkl_f:
                self._cached_shortest_dist = pickle.load(pkl_f)
        else:
            self._cached_shortest_dist = {}
        self._last_delta = 0.0

    def _compute_shortest_dist(self, episode: SequentialEpisode) -> float:
        if self._sim.habitat_config.TYPE == "ROS-Robot-v0":
            return self._cached_shortest_dist[episode.episode_id]
        else:
            distance = super()._compute_shortest_dist(episode)
            self._cached_shortest_dist[episode.episode_id] = distance
            return distance

    def __del__(self) -> None:
        if self._sim.habitat_config.TYPE != "ROS-Robot-v0":
            with open(CACHED_DATA_ROOT + "online_seq_spl.pkl", 'wb') as pkl_f:
                pickle.dump(self._cached_shortest_dist, pkl_f)

    def update_metric(self, episode: SequentialEpisode, task: SequentialNavigationTask,
                            *args: Any, **kwargs: Any) -> None:
        pos = np.array(self._sim.get_agent_state().position)
        if hasattr(self._sim, "intf_node"):
            self._last_delta = self._sim.intf_node.get_travel_distance_delta()
            self._cumul_dist += self._last_delta
        elif self._last_step_index == episode._current_step_index \
                and np.allclose(self._last_pos, pos):
                        return
        else:
            self._cumul_dist += np.linalg.norm(pos - self._last_pos)
        if task.measurements.measures[SequentialSuccess.cls_uuid].get_metric():
            self._metric = self._shortest_dist / max(self._shortest_dist, self._cumul_dist)
        self._last_pos = pos
        self._last_step_index = episode._current_step_index


@registry.register_measure
class OnlinePPL(PPL):
    _cached_shortest_dist: Dict[str, List[float]]

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        if self._sim.habitat_config.TYPE == "ROS-Robot-v0":
            with open(CACHED_DATA_ROOT + "online_ppl.pkl", 'rb') as pkl_f:
                self._cached_shortest_dist = pickle.load(pkl_f)
        else:
            self._cached_shortest_dist = {}

    def _compute_shortest_dist(self, episode: SequentialEpisode) -> List[float]:
        if self._sim.habitat_config.TYPE == "ROS-Robot-v0":
            return self._cached_shortest_dist[episode.episode_id]
        else:
            distances = super()._compute_shortest_dist(episode)
            self._cached_shortest_dist[episode.episode_id] = distances
            return distances

    def __del__(self) -> None:
        if self._sim.habitat_config.TYPE != "ROS-Robot-v0":
            with open(CACHED_DATA_ROOT + "online_ppl.pkl", 'wb') as pkl_f:
                pickle.dump(self._cached_shortest_dist, pkl_f)

    def update_metric(self, episode: SequentialEpisode, task: SequentialNavigationTask,
                            *args: Any, **kwargs: Any) -> None:
        pos = np.array(self._sim.get_agent_state().position)
        idx = episode._current_step_index
        if hasattr(self._sim, "intf_node"):
            if 'seq_spl' in task.measurements.measures:
                self._cumul_dist += task.measurements.measures['seq_spl']._last_delta
            else:
                self._cumul_dist += self._sim.intf_node.get_travel_distance_delta()
        elif self._last_step_index == idx and np.allclose(self._last_pos, pos):
            return
        else:
            self._cumul_dist += np.linalg.norm(pos - self._last_pos)
        if idx > 0:
            p = task.measurements.measures[Progress.cls_uuid].get_metric()
            d = self._shortest_dist[idx - 1]
            self._metric = p * d / max(self._cumul_dist, d)
        self._last_step_index = idx
        self._last_pos = pos