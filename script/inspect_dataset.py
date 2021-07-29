import enum
import itertools

import numpy as np
import quaternion
import cv2
import tqdm

import habitat
from habitat.utils.visualizations.maps import MAP_VALID_POINT
from habitat_sim2real import BaseSimulatorViewer


class DatasetInspector(BaseSimulatorViewer):
    class ShowAllModes(enum.Enum):
        PATHS = enum.auto()
        PATHS_HEATMAP = enum.auto()
        STARTS_HEATMAP = enum.auto()

    MAX_EP_ITER_FREQ = 50
    MIN_EP_ITER_FREQ = 1
    CV2_WAIT_TIME=1

    def __init__(self, cfg):
        data = habitat.make_dataset(cfg.DATASET.TYPE, config=cfg.DATASET)
        def scene_id(ep):
            return ep.scene_id
        episodes = sorted(data.episodes, key=scene_id)
        self.grouped_episodes = {k: list(grp)
                                 for k, grp in itertools.groupby(episodes, key=scene_id)}
        self.grp_iter = itertools.cycle(self.grouped_episodes.items())
        self.cur_scene, self.cur_scene_episodes = next(self.grp_iter)
        self.ep_iter = iter(self.cur_scene_episodes)
        self.cur_episode = next(self.ep_iter)

        self.pause_ep_iter = True
        self.ep_iter_freq = 20
        self.ep_time_elapsed = 0
        self.show_all_episodes = False
        self.show_all_mode_iter = itertools.cycle(DatasetInspector.ShowAllModes)
        self.cur_show_all_mode = next(self.show_all_mode_iter)
        self._cached_paths_heatmaps = {}

        cfg.defrost()
        cfg.SIMULATOR.SCENE = self.cur_scene
        cfg.freeze()

        super().__init__(cfg.SIMULATOR)

    def on_key(self, key_code):
        update = True
        if key_code == ord(' '):
            if not self.show_all_episodes:
                self.pause_ep_iter = not self.pause_ep_iter
        elif key_code == ord('b'):
            self.show_all_episodes = not self.show_all_episodes
            self.pause_ep_iter = True
        elif key_code == ord('h'):
            if self.show_all_episodes:
                self.cur_show_all_mode = next(self.show_all_mode_iter)
            else:
                self.show_all_episodes = True
            self.pause_ep_iter = True
        elif key_code == ord('n'):
            self.step_ep_iter()
        elif key_code == ord('+'):
            self.ep_iter_freq = min(self.MAX_EP_ITER_FREQ, self.ep_iter_freq + 1)
        elif key_code == ord('-'):
            self.ep_iter_freq = max(self.MIN_EP_ITER_FREQ, self.ep_iter_freq - 1)
        else:
            update = super().on_key(key_code)
        return update

    def step_ep_iter(self):
        try:
            self.cur_episode = next(self.ep_iter)
        except StopIteration:
            self.cur_scene, self.cur_scene_episodes = next(self.grp_iter)
            self.ep_iter = iter(self.cur_scene_episodes)
            self.cur_episode = next(self.ep_iter)

            self.cfg.defrost()
            self.cfg.SCENE = self.cur_scene
            self.cfg.freeze()
            self.sim.reconfigure(self.cfg)
        self.update()

    def time_update(self, dt):
        if self.pause_ep_iter:
            return
        self.ep_time_elapsed += dt
        if self.ep_time_elapsed >= 1000 / self.ep_iter_freq:
            self.step_ep_iter()
            self.ep_time_elapsed = 0
    
    def draw_episode_on_map(self, disp):
        if self.cur_episode._shortest_path_cache is None:
            self.sim.geodesic_distance(self.cur_episode.start_position,
                                       self.cur_episode.goals[0].position,
                                       self.cur_episode)
        points = np.array(self.cur_episode._shortest_path_cache.points)
        map_pts = self.project_pos_to_map(points)
        cv2.polylines(disp, [map_pts], False, (0, 215, 205), 2)

        start = self.project_pos_to_map(np.array(self.cur_episode.start_position))
        rot = np.quaternion(self.cur_episode.start_rotation[3],
                            *self.cur_episode.start_rotation[:3])
        q = np.quaternion(0, 0, 0, -1)
        start_head = (rot * q * rot.conjugate()).vec[::2]
        self.draw_agent_on_map(disp, start, start_head, (20, 205, 0))

        goal = self.project_pos_to_map(np.array(self.cur_episode.goals[0].position))
        cv2.circle(disp, tuple(goal), 5, (20, 0, 235), -1)

    def check_all_episodes_path_cache(self):
        for episode in self.cur_scene_episodes:
            if episode._shortest_path_cache is None:
                self.sim.geodesic_distance(episode.start_position,
                                           episode.goals[0].position,
                                           episode)

    def draw_all_episodes_paths(self, disp):
        self.check_all_episodes_path_cache()
        map_pts = [self.project_pos_to_map(np.array(episode._shortest_path_cache.points))
                   for episode in self.cur_scene_episodes]
        cv2.polylines(disp, map_pts, False, (0, 215, 205), 1)

    def draw_all_episodes_paths_heatmap(self, disp):
        if self.cur_scene in self._cached_paths_heatmaps:
            heatmap = self._cached_paths_heatmaps[self.cur_scene]
        else:
            self.check_all_episodes_path_cache()
            heatmap = np.zeros(self.raw_map.shape, dtype=np.float32)
            for episode in tqdm.tqdm(self.cur_scene_episodes, desc="Drawing paths heatmap"):
                ep_path = np.zeros_like(heatmap)
                points = np.array(episode._shortest_path_cache.points)
                map_pts = self.project_pos_to_map(points)
                cv2.polylines(ep_path, [map_pts], False, 1.0, 3)
                heatmap += ep_path
            heatmap /= heatmap.mean() + 3 * heatmap.std()
            heatmap = (255 * np.clip(heatmap, 0, 1)).astype(np.uint8)
            heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            self._cached_paths_heatmaps[self.cur_scene] = heatmap

        mask = self.raw_map == MAP_VALID_POINT
        disp[mask] = heatmap[mask]

    def draw_all_episodes_starts_heatmap(self, disp, expl_radius=3):
        heatmap = np.zeros(self.raw_map.shape, dtype=np.float32)
        points = np.array([episode.start_position for episode in self.cur_scene_episodes])
        map_pts = self.project_pos_to_map(points)
        for u, v in map_pts:
            heatmap[v, u] += 1.0

        sigma = expl_radius / (3 * self.map_resolution.mean())
        kern = cv2.getGaussianKernel(2 * int(5 * sigma) + 1, sigma)
        heatmap = cv2.sepFilter2D(heatmap, -1, kern, kern)
        heatmap /= heatmap.mean() + 3 * heatmap.std()
        heatmap = (255 * np.clip(heatmap, 0, 1)).astype(np.uint8)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

        mask = self.raw_map == MAP_VALID_POINT
        disp[mask] = heatmap[mask]

    def draw_map(self):
        disp = super().draw_map()
        if self.show_all_episodes:
            if self.cur_show_all_mode == DatasetInspector.ShowAllModes.PATHS:
                self.draw_all_episodes_paths(disp)
            elif self.cur_show_all_mode == DatasetInspector.ShowAllModes.PATHS_HEATMAP:
                self.draw_all_episodes_paths_heatmap(disp)
            elif self.cur_show_all_mode == DatasetInspector.ShowAllModes.STARTS_HEATMAP:
                self.draw_all_episodes_starts_heatmap(disp)
        else:
            self.draw_episode_on_map(disp)
        txt = "Ep#{}, iter: {}Hz".format(self.cur_episode.episode_id, self.ep_iter_freq)
        if self.pause_ep_iter:
            txt += " (paused)"
        cv2.putText(disp, txt, (0, 16), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 0))
        return disp


if __name__ == "__main__":
    import sys
    cfg = habitat.get_config("configs/locobot_pointnav_citi_sim.yaml", sys.argv[1:])
    viewer = DatasetInspector(cfg)
    viewer.run()
