import os
os.environ["GLOG_minloglevel"] = "2"
os.environ["MAGNUM_LOG"] = "quiet"
import logging

import numpy as np
from quaternion import rotate_vectors as quat_rot
import cv2

import habitat
habitat.logger.setLevel(logging.ERROR)

from habitat_sim import NavMeshSettings
from habitat.utils.visualizations.maps import get_topdown_map_from_sim, colorize_topdown_map


CFG_PATH = "configs/locobot_pointnav_gibson.yaml"
SCENE_PATH = "data/scene_datasets/nle/09c52b4f283344bf92e4a71df646c9b4.glb"
NAVMESH_SETTINGS_BOUNDS = {"agent_max_slope": (5, 60), "agent_max_climb": (0.0, 0.50)}


def draw_agent(disp, pos, vec, color):
    cv2.circle(disp, tuple(pos), 5, color, -1)
    cv2.line(disp, tuple(pos), tuple((pos + 10 * vec).astype(np.int64)), color, 3)


class NavmeshInspector:
    def __init__(self, cfg_path, scene_path, win_name="Inspector", scale=None):
        print("Reading config...", end=' ', flush=True)
        self.cfg = habitat.get_config(CFG_PATH).SIMULATOR
        self.cfg.defrost()
        self.cfg.SCENE = SCENE_PATH
        self.cfg.freeze()
        print("Done!")

        print("Loading simulator...", end=' ', flush=True)
        self.sim = habitat.sims.make_sim(self.cfg.TYPE, config=self.cfg)
        self.sim.navmesh_visualization = True
        self.obs = self.sim.reset()
        self.navmesh_settings = NavMeshSettings()
        self.navmesh_settings.set_defaults()
        print("Done!")

        self.win_name = win_name
        cv2.namedWindow(self.win_name)
        cv2.setMouseCallback(self.win_name, self.on_mouse)
        self.obs_win_name = win_name + " - Observations"
        cv2.namedWindow(self.obs_win_name)

        print("Generating map...", end=' ', flush=True)
        self.map_img = colorize_topdown_map(get_topdown_map_from_sim(self.sim))

        self.scale = scale
        if self.scale:
            self.map_img = cv2.resize(self.map_img, None, fx=self.scale, fy=self.scale)

        bounds = self.sim.pathfinder.get_bounds()
        self.map_origin = np.array([bounds[0][0], bounds[0][2]])
        self.map_resolution = np.array([(bounds[1][0] - bounds[0][0]) / self.map_img.shape[1],
                                        (bounds[1][2] - bounds[0][2]) / self.map_img.shape[0]])
        print("Done!")

        self.drag_start = None
        self.drag_vec = None

    def on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drag_start = np.array([x, y]).astype(np.int64)
        elif event == cv2.EVENT_MOUSEMOVE and self.drag_start is not None:
            drag_vec = np.array([x, y]).astype(np.float32) - self.drag_start
            norm = np.sqrt(drag_vec.dot(drag_vec))
            if norm > 0:
                self.drag_vec = drag_vec / norm
        elif event == cv2.EVENT_LBUTTONUP:
            self.teleport_agent()
            self.drag_start = None
            self.drag_vec = None

    def on_key_press(self, key_code):
        if key_code == ord('s'):
            return True
        elif key_code == ord('w'):
            self.obs = self.sim.step(1)
        elif key_code == ord('a'):
            self.obs = self.sim.step(2)
        elif key_code == ord('d'):
            self.obs = self.sim.step(3)
        elif key_code == ord('u'):
            self.update_navmesh("agent_max_climb", +0.02)
        elif key_code == ord('j'):
            self.update_navmesh("agent_max_climb", -0.02)
        elif key_code == ord('i'):
            self.update_navmesh("agent_max_slope", +5)
        elif key_code == ord('k'):
            self.update_navmesh("agent_max_slope", -5)
        return False

    def teleport_agent(self):
        pos = self.drag_start * self.map_resolution + self.map_origin
        s = self.sim.get_agent_state()
        robot_y = s.position[1]
        pos = np.array([pos[0], robot_y, pos[1]], dtype=np.float32)

        if self.drag_vec is None:
            rot = [0.0, 1.0, 0.0, 0.0]
        else:
            yaw = np.pi + np.arctan2(self.drag_vec[0], self.drag_vec[1])
            rot = [0, np.sin(0.5 * yaw), 0, np.cos(0.5 * yaw)]
        self.obs = self.sim.get_observations_at(pos, rot, True)

    def update_navmesh(self, attr, mod):
        prv_val = getattr(self.navmesh_settings, attr)
        val_min, val_max = NAVMESH_SETTINGS_BOUNDS[attr]
        val = max(val_min, min(prv_val + mod, val_max))
        if val != prv_val:
            setattr(self.navmesh_settings, attr, val)
            print("Updating navmesh using '{}'={:.2f}...".format(attr, val), end=' ', flush=True)
            success = self.sim.recompute_navmesh(self.sim.pathfinder, self.navmesh_settings)
            print("Done!" if success else "FAIL!!")
            if success:
                self.obs = self.sim.get_observations_at()
                print("Generating map...", end=' ', flush=True)
                self.map_img = colorize_topdown_map(get_topdown_map_from_sim(self.sim))
                if self.scale:
                    self.map_img = cv2.resize(self.map_img, None, fx=self.scale, fy=self.scale)
                print("Done!")

    def draw_map(self):
        disp = self.map_img.copy()
        s = self.sim.get_agent_state()
        map_pos = ((s.position[::2] - self.map_origin) / self.map_resolution).astype(np.int64)
        head = quat_rot(s.rotation, self.sim.forward_vector[None, :])[0, ::2]
        draw_agent(disp, map_pos, head, (255, 0, 0))
        if self.drag_vec is not None:
            draw_agent(disp, self.drag_start, self.drag_vec, (0, 255, 0))
        return disp
    
    def draw_obs(self):
        disp_depth = cv2.cvtColor((self.obs["depth"] * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        disp = np.vstack((self.obs["rgb"][:, :, ::-1], disp_depth))
        if self.scale:
            disp = cv2.resize(disp, None, fx=self.scale, fy=self.scale)
        return disp

    def show(self):
        cv2.imshow(self.win_name, self.draw_map())
        cv2.imshow(self.obs_win_name, self.draw_obs())

    def run(self):
        while True:
            self.show()
            c = cv2.waitKey(30)
            if self.on_key_press(c):
                break


if __name__ == "__main__":
    inspect = NavmeshInspector(CFG_PATH, SCENE_PATH, scale=0.8)
    inspect.run()
