import os
os.environ["GLOG_minloglevel"] = "2"
os.environ["MAGNUM_LOG"] = "quiet"
import logging
import argparse
import gzip
import itertools

import numpy as np
import cv2
import tqdm

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


class ColorNavMapGUI:
    def __init__(self, sim_cfg, map_height=None, map_resolution=0.03, name="Map"):
        print("Loading navmap from simulator...")
        with habitat.sims.make_sim(sim_cfg.TYPE, config=sim_cfg) as sim:
            if map_height is None:
                map_height = sim.get_agent_state().position[1]
            self.scene_id = sim_cfg.SCENE
            self.navmask = sim.pathfinder.get_topdown_view(map_resolution, map_height)
            self.origin, _ = sim.pathfinder.get_bounds()
            self.origin[1] = map_height

        self.edges = np.zeros_like(self.navmask)
        self.edges[:-1, :-1] = (self.navmask[:-1, :-1] != self.navmask[1:, :-1]) \
                             | (self.navmask[:-1, :-1] != self.navmask[:-1, 1:])
        self.zones = np.zeros_like(self.navmask, dtype=np.uint8)
        self.zones[self.navmask] = 1
        self.last_idx = 2
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
            self.select_history.append(self.zones.copy())
            self.drag_start = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and self.drag_start is not None:
            self.drag_end = (x, y)
            self.draw()
        elif event == cv2.EVENT_LBUTTONUP:
            self.update_zones()
            self.drag_start = None
            self.drag_end = None
            self.draw()

    def on_key_press(self, key_code):
        if key_code == ord('u'):
            if self.select_history:
                self.zones = self.select_history.pop()
            elif self.confirm_history:
                self.zones = self.confirm_history.pop()
                self.last_idx -= 1
            self.draw()
        elif key_code == ord('c') and self.select_history:
            self.last_idx += 1
            self.confirm_history.append(self.select_history[0])
            self.select_history = []
            if self.last_idx > 255:
                raise ValueError("Too many zones! Can only use 254 different ones.")
            self.draw()
        elif key_code == ord('x'):
            self.do_loop = False

    def update_zones(self):
        l, r = sorted((self.drag_start[0], self.drag_end[0]))
        t, b = sorted((self.drag_start[1], self.drag_end[1]))
        mask = self.navmask[t:b, l:r]
        self.zones[t:b, l:r][mask] = self.last_idx

    def get_zone_center(self, lbl):
        i, j = np.nonzero(self.zones == lbl)
        return int(j.mean()), int(i.mean())

    def draw(self):
        disp = self.colors[self.zones]
        for lbl in range(1, self.last_idx):
            x, y = self.get_zone_center(lbl)
            c = (self.colors[lbl] // 2).tolist()
            cv2.putText(disp, str(lbl), (x, y), cv2.FONT_HERSHEY_COMPLEX, 1, c, 2)
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
        return self.zones

    def close(self):
        cv2.destroyWindow(self.name)

COL0_W = 96
COL_W = 32
COL_SEP = 4
ROW_H = 32
ROW_SEP = 4


class SplitZonesGUI:
    def __init__(self, zone_colors, name="Split"):
        self.colors = zone_colors
        self.num_zones = zone_colors.shape[0]
        self.splits = [[True for _ in range(self.num_zones)]]
        self.num_splits = 1
        self.auto_name_gen = (f"Split{i}" for i in itertools.count())
        self.labels = [next(self.auto_name_gen)]
        self.renaming = -1
        self.prv_name = ""

        self.name = name
        cv2.namedWindow(self.name, cv2.WINDOW_GUI_NORMAL | cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(self.name, self.on_mouse_event)
        self.do_loop = True

    def on_mouse_event(self, event, x, y, flags, params):
        if event == cv2.EVENT_LBUTTONDOWN and self.renaming < 0:
            if y > (self.num_splits + 1) * (ROW_H + ROW_SEP):
                if x < (self.w - COL_SEP) // 2:
                    self.add_split()
                elif x >= (self.w + COL_SEP) // 2:
                    self.do_loop = False
            elif y > ROW_H + ROW_SEP:
                s = (y - (ROW_H + ROW_SEP)) // (ROW_H + ROW_SEP)
                if x > COL0_W + COL_SEP + (self.num_zones + 1) * (COL_W + COL_SEP):
                    self.delete_split(s)
                elif x > COL0_W + COL_SEP + self.num_zones * (COL_W + COL_SEP):
                    self.complement_split(s)
                elif x > COL0_W + COL_SEP:
                    z = (x - (COL0_W + COL_SEP)) // (COL_W + COL_SEP)
                    self.splits[s][z] = not self.splits[s][z]
                else:
                    self.prv_name = self.labels[s]
                    self.labels[s] = ""
                    self.renaming = s
            self.draw()

    def add_split(self):
        self.splits.append([True for _ in range(self.num_zones)])
        self.labels.append(next(self.auto_name_gen))
        self.num_splits += 1

    def delete_split(self, s):
        del self.splits[s]
        del self.labels[s]
        self.num_splits -= 1

    def complement_split(self, s):
        self.splits.insert(s + 1, [not zone_in_split for zone_in_split in self.splits[s]])
        self.labels.insert(s + 1, f"Not{self.labels[s]}")
        self.num_splits += 1

    def draw(self):
        self.h = ROW_H * (self.num_splits + 2) + ROW_SEP * (self.num_splits + 1)
        self.w = COL0_W + (COL_W + COL_SEP) * (self.num_zones + 2)
        disp = np.full((self.h, self.w, 3), 127, dtype=np.uint8)
        for c, color in enumerate(self.colors):
            l = COL0_W + (c + 1) * COL_SEP + c * COL_W
            disp[:ROW_H, l:l + COL_W] = color
            txt_color = (color // 2).tolist()
            cv2.putText(disp, str(c + 1), (l + COL_W // 5, 3 * ROW_H // 4),
                        cv2.FONT_HERSHEY_COMPLEX, 1, txt_color, 2)
        for s, name in enumerate(self.labels):
            t = (s + 1) * (ROW_H + ROW_SEP)
            disp[t:t + ROW_H, :COL0_W] = 255
            cv2.putText(disp, name, (0, t + 3 * ROW_H // 4),
                        cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 0, 0), 1)
            for z in range(self.num_zones):
                l = COL0_W + (z + 1) * COL_SEP + z * COL_W
                disp[t:t + ROW_H, l:l + COL_W] = 255
                cv2.putText(disp, "+" if self.splits[s][z] else "-",
                            (l + COL_W // 5, t + 3 * ROW_H // 4),
                            cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 0, 0), 1)
            l = COL0_W + (self.num_zones + 1) * COL_SEP + self.num_zones * COL_W
            disp[t:t + ROW_H, l:l + COL_W] = 255
            cv2.putText(disp, "C", (l + COL_W // 5, t + 3 * ROW_H // 4),
                        cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 0, 0), 1)
            l = COL0_W + (self.num_zones + 2) * COL_SEP + (self.num_zones + 1) * COL_W
            disp[t:t + ROW_H, l:l + COL_W] = 255
            cv2.putText(disp, "x", (l + COL_W // 5, t + 3 * ROW_H // 4),
                        cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 0, 0), 1)

        t = (self.num_splits + 1) * (ROW_H + ROW_SEP)
        disp[t:t + ROW_H, :(self.w - COL_SEP) // 2] = 255
        cv2.putText(disp, "Add split", (COL_W, t + 3 * ROW_H // 4),
                    cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 0, 0), 1)
        disp[t:t + ROW_H, (self.w + COL_SEP) // 2:] = 255
        cv2.putText(disp, "Done!", (self.w // 2 + COL_W, t + 3 * ROW_H // 4),
                    cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 0, 0), 1)
        cv2.imshow(self.name, disp)

    def loop(self):
        self.draw()
        while self.do_loop:
            c = cv2.waitKey(1)
            if self.renaming >= 0:
                if c == ord('\r'):
                    if len(self.labels[self.renaming]) == 0:
                        self.labels[self.renaming] = self.prv_name
                    self.renaming = -1
                elif ord('A') <= c <= ord('z'):
                    self.labels[self.renaming] = self.labels[self.renaming] + chr(c)
                self.draw()
        return [{i for i, zone_in_split in enumerate(split, start=1) if not zone_in_split}
                for split in self.splits]

    def close(self):
        cv2.destroyWindow(self.name)


def main(args):
    habitat.logger.setLevel(logging.ERROR)
    cfg = habitat.get_config(args.config_path, args.extra_cfg)
    color_gui = ColorNavMapGUI(cfg.SIMULATOR, args.map_height, args.map_resolution)
    zones = color_gui.loop()

    split_gui = SplitZonesGUI(color_gui.colors[1:color_gui.last_idx])
    splits = split_gui.loop()

    print("Loading source dataset...")
    data_in = habitat.make_dataset(cfg.DATASET.TYPE, config=cfg.DATASET)
    data_out = [habitat.make_dataset(cfg.DATASET.TYPE) for _ in splits]
    for episode in tqdm.tqdm(data.episodes):
        xyz = np.array(sim.get_straight_shortest_path_points(episode.start_position,
                                                             episode.goals[0].position))
        j_i = ((xyz - origin) / args.map_resolution).astype(np.int64)
        path_mask = np.zeros(zones.shape, dtype=np.uint8)
        cv2.polylines(path_mask, j_i[None, :, [0, 2]], False, 1)
        path_mask = path_mask.astype(np.bool)
        episode_zones = set(zones[path_mask])

        for data_split, excluded_zones in zip(data_out, splits):
            if episode_zones & excluded_zones:
                continue
            data_split.episodes.append(episode)

    data_path = cfg.DATASET.DATA_PATH.format(split=cfg.DATASET.SPLIT)
    out_ext = ".json.gz"
    out_prefix = data_path[:-len(out_ext)]
    for name, data_split in zip(split_gui.labels, data_out):
        out_path = out_prefix + f"_{name}" + out_ext
        with gzip.open(out_path, 'wt') as f:
            f.write(data_split.to_json())


if __name__ == "__main__":
    main(parse_args())
