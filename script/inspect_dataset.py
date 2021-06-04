import sys
import cv2
from quaternion import quaternion as quat
import habitat
from habitat_sim2real import BaseSimulatorViewer


class DatasetInspector(BaseSimulatorViewer):
    MAX_EP_ITER_FREQ = 50
    MIN_EP_ITER_FREQ = 1
    CV2_WAIT_TIME=1

    def __init__(self, cfg):
        dataset = habitat.make_dataset(cfg.DATASET.TYPE, config=cfg.DATASET)
        self.ep_iter = dataset.get_episode_iterator(cycle=True, group_by_scene=True)
        self.cur_episode = next(self.ep_iter)
        self.pause_ep_iter = True
        self.ep_iter_freq = 20
        self.ep_time_elapsed = 0

        cfg.defrost()
        cfg.SIMULATOR.SCENE = self.cur_episode.scene_id
        cfg.freeze()

        super().__init__(cfg.SIMULATOR)

    def on_key(self, key_code):
        update = True
        if key_code == ord(' '):
            self.pause_ep_iter = not self.pause_ep_iter
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
        self.cur_episode = next(self.ep_iter)
        if self.cfg.SCENE != self.cur_episode.scene_id:
            self.cfg.defrost()
            self.cfg.SCENE = self.cur_episode.scene_id
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
        points = self.cur_episode._shortest_path_cache.points
        for pt1, pt2 in zip(points, points[1:]):
            map_pt1 = self.project_pos_to_map(pt1)
            map_pt2 = self.project_pos_to_map(pt2)
            cv2.line(disp, tuple(map_pt1), tuple(map_pt2), (0, 215, 205), 2)
        start = self.project_pos_to_map(self.cur_episode.start_position)
        rot = quat(self.cur_episode.start_rotation[3], *self.cur_episode.start_rotation[:3])
        q = quat(0, 0, 0, -1)
        start_head = (rot * q * rot.conjugate()).vec[::2]
        self.draw_agent_on_map(disp, start, start_head, (20, 205, 0))
        goal = self.project_pos_to_map(self.cur_episode.goals[0].position)
        cv2.circle(disp, tuple(goal), 5, (20, 0, 235), -1)

    def draw_map(self):
        disp = super().draw_map()
        self.draw_episode_on_map(disp)
        txt = "Ep#{}, iter: {}Hz".format(self.cur_episode.episode_id, self.ep_iter_freq)
        if self.pause_ep_iter:
            txt += " (paused)"
        cv2.putText(disp, txt, (0, 16), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 0))
        return disp


if __name__ == "__main__":
    cfg = habitat.get_config("configs/locobot_pointnav_citi_sim.yaml", sys.argv[1:])
    viewer = DatasetInspector(cfg)
    viewer.run()
