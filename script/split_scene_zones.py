import os
os.environ["GLOG_minloglevel"] = "2"
os.environ["MAGNUM_LOG"] = "quiet"
import logging
import argparse
import gzip
import json

import tqdm
import numpy as np
import cv2

import habitat


DEFAULT_ARGS = {"config_path": "configs/locobot_pointnav_citi_sim.yaml",
                "map_height": None,
                "map_resolution": 0.03}


def parse_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config-path", "-c")
    parser.add_argument("--map-height", "-h", type=float)
    parser.add_argument("--map-resolution", "-r", type=float)
    parser.add_argument("--help", "-?", action="help")
    parser.add_argument("extra_cfg", nargs=argparse.REMAINDER)
    parser.set_defaults(**DEFAULT_ARGS)
    return parser.parse_args()


class NavGUI:
    def __init__(self, mask, name="Map"):
        self.mask = mask
        self.edges = np.zeros_like(mask)
        self.edges[:-1, :-1] = (mask[:-1, :-1] != mask[1:, :-1]) \
                             | (mask[:-1, :-1] != mask[:-1, 1:])
        self.labels = np.zeros_like(mask, dtype=np.uint8)
        self.labels[mask] = 1
        self.last_lbl = 2
        self.confirm_history = []
        self.select_history = []

        no_lbl_colors = np.array([[157, 157, 157],
                                  [255, 255, 255]], dtype=np.uint8)
        color_pattern = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1],
                                  [1, 1, 0], [1, 0, 1], [0, 1, 1]], dtype=np.uint8)
        self.colors = np.concatenate((no_lbl_colors,
                                      *(color_pattern * 255 // k
                                        for k in range(1, 44))), 0)

        self.name = name
        cv2.namedWindow(self.name, cv2.WINDOW_GUI_NORMAL | cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(self.name, self.on_mouse_event)
        self.drag_start = None
        self.drag_end = None
        self.new_selection = True
        self.do_loop = True

    def on_mouse_event(self, event, x, y, flags, params):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.select_history.append(self.labels.copy())
            self.drag_start = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and self.drag_start is not None:
            self.drag_end = (x, y)
            self.draw()
        elif event == cv2.EVENT_LBUTTONUP:
            self.update_labels()
            self.drag_start = None
            self.drag_end = None
            self.draw()

    def on_key_press(self, key_code):
        if key_code == ord('u'):
            if self.select_history:
                self.labels = self.select_history.pop()
            elif self.confirm_history:
                self.labels = self.confirm_history.pop()
                self.last_lbl -= 1
            self.draw()
        elif key_code == ord('c') and self.select_history:
            self.last_lbl += 1
            self.confirm_history.append(self.select_history[0])
            self.select_history = []
            if self.last_lbl > 255:
                raise ValueError("Too many labels! Can only use 254 different ones.")
        elif key_code == ord('x'):
            self.do_loop = False

    def update_labels(self):
        l, r = sorted((self.drag_start[0], self.drag_end[0]))
        t, b = sorted((self.drag_start[1], self.drag_end[1]))
        mask = self.mask[t:b, l:r]
        self.labels[t:b, l:r][mask] = self.last_lbl

    def draw(self):
        disp = self.colors[self.labels]
        if self.drag_end is not None:
            cv2.rectangle(disp, self.drag_start, self.drag_end, (0, 0, 0))
        disp[self.edges] = 0
        cv2.imshow(self.name, disp)
        return disp

    def loop(self):
        self.draw()
        while self.do_loop:
            c = cv2.waitKey(1)
            if c > -1:
                self.on_key_press(c)
        return self.labels

    def close(self):
        cv2.destroyWindow(self.name)


def main(args):
    habitat.logger.setLevel(logging.ERROR)
    cfg = habitat.get_config(args.config_path, args.extra_cfg)
    data = habitat.make_dataset(cfg.DATASET.TYPE, config=cfg.DATASET)
    with habitat.sims.make_sim(cfg.SIMULATOR.TYPE, config=cfg.SIMULATOR) as sim:
        if args.map_height is None:
            args.map_height = sim.get_agent_state().position[1]

        mask = sim.pathfinder.get_topdown_view(args.map_resolution, args.map_height)
        origin, _ = sim.pathfinder.get_bounds()
        origin[1] = args.map_height

        gui = NavGUI(mask)
        labels = gui.loop()
        gui.close()

        splits = [habitat.make_dataset(cfg.DATASET.TYPE) for _ in range(1, gui.last_lbl)]
        for episode in tqdm.tqdm(data.episodes):
            xyz = np.array(sim.get_straight_shortest_path_points(episode.start_position,
                                                                 episode.goals[0].position))
            j_i = ((xyz - origin) / args.map_resolution).astype(np.int64)
            path_mask = np.zeros(mask.shape, dtype=np.uint8)
            cv2.polylines(path_mask, j_i[None, :, [0, 2]], False, 1)
            path_mask = path_mask.astype(np.bool)
            ep_labels = set(labels[path_mask]) - {0}
            for lbl, split in enumerate(splits, start=1):
                if lbl not in ep_labels:
                    split.episodes.append(episode)

        data_path = cfg.DATASET.DATA_PATH.format(split=cfg.DATASET.SPLIT)
        out_ext = ".json.gz"
        out_prefix = data_path[:-len(out_ext)]
        for lbl, split in enumerate(splits, start=1):
            out_path = out_prefix + f"_zone{lbl}_excluded" + out_ext
            with gzip.open(out_path, 'wt') as f:
                json.dump(split.to_json(), f)


if __name__ == "__main__":
    main(parse_args())
