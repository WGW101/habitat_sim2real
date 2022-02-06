from typing import List, Any
import pickle
from habitat.tasks.sequential_nav.sequential_nav import SequentialEpisode, SequentialSPL


CACHED_DATA_ROOT = "data/cached_measurements/"


@registry.register_measure
class OnlineSequentialSPL(SequentialSPL):
    _cached_shortest_dist: Dict[str, float]

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        if self._sim.habitat_config.TYPE == "ROS-Robot-v0":
            with open(CACHED_DATA_ROOT + "online_seq_spl.pkl", 'rb') as pkl_f:
                self._cached_shortest_dist = pickle.load(pkl_f)
        else:
            self._cached_shortest_dist = {}

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
