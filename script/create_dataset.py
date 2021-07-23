import gzip
import itertools
import os
import sys
import habitat
from habitat.tasks.nav.nav import NavigationGoal, NavigationEpisode
from habitat_sim2real import BaseSimulatorViewer


class DatasetInspector(BaseSimulatorViewer):
    def __init__(self, cfg):
        self.dataset_type = cfg.DATASET.TYPE
        self.out_path = cfg.DATASET.DATA_PATH.format(split=cfg.DATASET.SPLIT)
        os.makedirs(os.path.dirname(self.out_path), exist_ok=True)
        self.episodes = []
        self.ep_id_gen = (f"episode{i:04d}" for i in itertools.count())
        self.start_position = None
        self.start_rotation = None
        self.auto_save = False
        super().__init__(cfg.SIMULATOR)

    def on_key(self, key_code):
        update = True
        if key_code == ord(' '):
            self.store_position()
        elif key_code == ord('b'):
            self.save_last_episode()
        elif key_code == ord('n'):
            self.save_dataset()
        elif key_code == ord('x'):
            if self.auto_save:
                self.save_dataset()
            self.running = False
        elif key_code == ord('v'):
            self.auto_save = not self.auto_save
            print("Auto save toggled {}".format("on" if self.auto_save else "off"))
        else:
            update = super().on_key(key_code)
        return update

    def store_position(self):
        s = self.sim.get_agent_state()
        if self.start_position is None:
            self.start_position = s.position.tolist()
            self.start_rotation = [s.rotation.x, s.rotation.y, s.rotation.z, s.rotation.w]
            print("Start position:", self.start_position)
            print("Start rotation:", self.start_rotation)
        else:
            goal = NavigationGoal(position=s.position.tolist())
            print("Goal position:", goal.position)
            episode = NavigationEpisode(episode_id=next(self.ep_id_gen),
                                        scene_id=self.cfg.SCENE,
                                        start_position=self.start_position,
                                        start_rotation=self.start_rotation,
                                        goals=[goal])
            self.episodes.append(episode)
            print(f"Episode '{episode.episode_id}' added to dataset")
            self.start_position = None
            self.start_rotation = None
            if self.auto_save:
                self.save_last_episode()

    def save_last_episode(self):
        if not self.episodes:
            return
        dataset = habitat.make_dataset(self.dataset_type)
        dataset.episodes = self.episodes[-1:]
        out_ext = ".json.gz"
        out_path = self.out_path[:-len(out_ext)] \
                + "_" + self.episodes[-1].episode_id + out_ext
        with gzip.open(out_path, 'wt') as f:
            f.write(dataset.to_json())
        print(f"Last episode saved to: {out_path}")

    def save_dataset(self):
        if not self.episodes:
            return
        dataset = habitat.make_dataset(self.dataset_type)
        dataset.episodes = self.episodes
        with gzip.open(self.out_path, 'wt') as f:
            f.write(dataset.to_json())
        print(f"Dataset saved to: {self.out_path}")


if __name__ == "__main__":
    cfg = habitat.get_config("configs/locobot_pointnav_citi_sim.yaml", sys.argv[1:])
    viewer = DatasetInspector(cfg)
    viewer.run()

