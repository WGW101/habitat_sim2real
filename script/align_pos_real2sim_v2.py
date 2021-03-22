import os
os.environ["GLOG_minloglevel"] = "2"
os.environ["MAGNUM_LOG"] = "quiet"
import logging
import habitat
habitat.logger.setLevel(logging.ERROR)
from habitat_sim2real import BaseSimulatorViewer

import numpy as np
import quaternion as quat
import cv2


class AlignViewer(BaseSimulatorViewer):
    def __init__(self, cfg, real_obs_path, draw_grid=True):
        super().__init__(cfg, win_basename="Align")

        self.out_path = real_obs_path.replace("real", "sim")
        os.makedirs(self.out_path, exist_ok="True")
        self.match_file = open(os.path.join(self.out_path, "match.csv"), 'wt')
        self.match_file.write("#REAL RGB PATH, #SIM RGB PATH\n")
        with open(os.path.join(real_obs_path, "chrono_list.txt")) as f:
            self.real_paths = [l.strip() for l in f]
        poses = np.array([[float(s[1:]) for s in os.path.basename(path).split('_')[:4]]
                          for path in self.real_paths])
        self.real_positions = poses[:, :3]
        a = np.radians(poses[:, 3])
        rotations = np.zeros((a.shape[0], 4))
        rotations[:, 0] = np.cos(0.5 * a)
        rotations[:, 3] = np.sin(0.5 * a)
        self.real_rotations = quat.from_float_array(rotations)

        self.draw_grid = draw_grid

        self.pos_offset = None
        self.rot_offset = None

    def on_key_press(self, key_code):
        update = True
        if key_code == ord('i'):
            self.obs = self.sim.step(4)
        elif key_code == ord('k'):
            self.obs = self.sim.step(5)
        elif key_code == ord('c'):
            self.accept_alignment()
        else:
            update = super().on_key_press(key_code)
        return update

    def accept_alignment(self):
        self.draw_agent_on_map(self.map_img, color=(120, 40, 30))
        self.save_images()
        self.calc_offset()
        self.running = False

    def calc_offset(self):
        ax_rot = quat.quaternion(0.5, -0.5, 0.5, 0.5)
        s = self.sim.get_agent_state()

        self.rot_offset = ax_rot * self.cur_rot.conj() * ax_rot.conj() * s.rotation
        pos_q = self.rot_offset * ax_rot \
                * quat.quaternion(0, *self.cur_pos) \
                * ax_rot.conj() * self.rot_offset.conj()
        self.pos_offset = s.position - pos_q.vec
        print("Offset:", self.pos_offset,
                         np.degrees(2 * np.arctan(self.rot_offset.y/self.rot_offset.w)))

    def save_images(self):
        s = self.sim.get_agent_state()
        a = np.degrees(2 * np.arctan(s.rotation.y / s.rotation.w))
        basename = "x{:.3f}_y{:.3f}_z{:.3f}_r{:.0f}_rgb.jpeg".format(*s.position, a)
        path = os.path.join(self.out_path, basename)
        cv2.imwrite(path, self.obs["rgb"][:, :, ::-1])
        converted = (255 * self.obs["depth"]).astype(np.uint8)
        cv2.imwrite(path.replace("rgb", "depth"), converted)
        self.match_file.write("{}, {}\n".format(self.cur_path, path))

    def apply_offset(self):
        ax_rot = quat.quaternion(0.5, -0.5, 0.5, 0.5)
        pos_q = self.rot_offset * ax_rot \
                * quat.quaternion(0, *self.cur_pos) \
                * ax_rot.conj() * self.rot_offset.conj()
        rot = self.rot_offset * ax_rot \
                * self.cur_rot \
                * ax_rot.conj() * self.rot_offset.conj()
        self.obs = self.sim.get_observations_at(self.pos_offset + pos_q.vec,
                                                self.rot_offset * rot, True)

    def get_real_observations(self):
        yield from zip(self.real_paths, self.real_positions, self.real_rotations)

    def draw_obs(self):
        disp = np.vstack((self.obs["rgb"][:, :, ::-1], self.real_rgb))
        if self.scale:
            disp = cv2.resize(disp, None, fx=self.scale, fy=self.scale)
        if self.draw_grid:
            h, w, _ = disp.shape
            cv2.line(disp, (w // 2, 0), (w // 2, h), (0, 0, 255))
            for i in (1, 3):
                cv2.line(disp, (0, i * h // 4), (w, i * h // 4), (0, 0, 255))
                cv2.line(disp, (i * w // 4, 0), (i * w // 4, h), (0, 255, 0))
            for i in range(1, 8, 2):
                cv2.line(disp, (0, i * h // 8), (w, i * h // 8), (0, 255, 0))
                cv2.line(disp, (i * w // 8, 0), (i * w // 8, h), (255, 0, 0))
            for i in range(1, 16, 2):
                cv2.line(disp, (0, i * h // 16), (w, i * h // 16), (255, 0, 0))
        return disp

    def run(self):
        for path, pos, rot in self.get_real_observations():
            self.cur_path = path
            self.cur_pos = pos
            self.cur_rot = rot
            self.real_rgb = cv2.imread(path)
            if self.pos_offset is not None:
                self.apply_offset()
            super().run()


if __name__ == "__main__":
    cfg = habitat.get_config("configs/locobot_pointnav_citi_sim.yaml").SIMULATOR
    cfg.defrost()
    cfg.FORWARD_STEP_SIZE = 0.05
    cfg.TURN_ANGLE = 1
    cfg.TILT_ANGLE = 1
    cfg.freeze()

    viewer = AlignViewer(cfg, "out/traj_cap/21-03-18/real")
    viewer.run()
