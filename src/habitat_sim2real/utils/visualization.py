import numpy as np
from quaternion import rotate_vectors as quat_rot
import cv2

import habitat
from habitat.utils.visualizations.maps import get_topdown_map_from_sim, colorize_topdown_map


class BaseSimulatorViewer:
    def __init__(self, sim_cfg, win_basename="Simulator", scale=None):
        self.sim = habitat.sims.make_sim(sim_cfg.TYPE, config=sim_cfg)
        self.obs = self.sim.reset()

        self.map_win_name = win_basename + " - Map"
        cv2.namedWindow(self.map_win_name)
        cv2.setMouseCallback(self.map_win_name, self.on_mouse)
        self.obs_win_name = win_basename + " - Observations"
        cv2.namedWindow(self.obs_win_name)

        self.map_img = colorize_topdown_map(get_topdown_map_from_sim(self.sim))
        self.scale = scale
        if self.scale:
            self.map_img = cv2.resize(self.map_img, None, fx=self.scale, fy=self.scale)
        bounds = self.sim.pathfinder.get_bounds()
        self.map_origin = np.array([bounds[0][0], bounds[0][2]])
        self.map_resolution = np.array([(bounds[1][0] - bounds[0][0]) / self.map_img.shape[1],
                                        (bounds[1][2] - bounds[0][2]) / self.map_img.shape[0]])

        self.drag_start = None
        self.drag_vec = None
        self.running = True

    def on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drag_start = np.array([x, y]).astype(np.int64)
        elif event == cv2.EVENT_MOUSEMOVE and self.drag_start is not None:
            drag_vec = np.array([x, y]).astype(np.float32) - self.drag_start
            norm = np.sqrt(drag_vec.dot(drag_vec))
            if norm > 0:
                self.drag_vec = drag_vec / norm
                self.update()
        elif event == cv2.EVENT_LBUTTONUP:
            self.teleport_agent()
            self.drag_start = None
            self.drag_vec = None
            self.update()

    def on_key_press(self, key_code):
        update = True
        if key_code == ord('s'):
            self.running = False
        elif key_code == ord('w'):
            self.obs = self.sim.step(1)
        elif key_code == ord('a'):
            self.obs = self.sim.step(2)
        elif key_code == ord('d'):
            self.obs = self.sim.step(3)
        elif key_code == ord('r'):
            self.obs = self.sim.reset()
        else:
            update = False
        return update

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

    def draw_map(self):
        disp = self.map_img.copy()
        s = self.sim.get_agent_state()
        map_pos = ((s.position[::2] - self.map_origin) / self.map_resolution).astype(np.int64)
        head = quat_rot(s.rotation, self.sim.forward_vector[None, :])[0, ::2]
        cv2.circle(disp, tuple(map_pos), 5, (255, 0, 0), -1)
        end = (map_pos + 10 * head).astype(np.int64)
        cv2.line(disp, tuple(map_pos), tuple(end), (255, 0, 0), 3)
        if self.drag_vec is not None:
            cv2.circle(disp, tuple(self.drag_start), 5, (0, 255, 0), -1)
            end = (self.drag_start + 10 * self.drag_vec).astype(np.int64)
            cv2.line(disp, tuple(self.drag_start), tuple(end), (0, 255, 0), 3)
        return disp
    
    def draw_obs(self):
        disp_depth = cv2.cvtColor((self.obs["depth"] * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        disp = np.vstack((self.obs["rgb"][:, :, ::-1], disp_depth))
        if self.scale:
            disp = cv2.resize(disp, None, fx=self.scale, fy=self.scale)
        return disp

    def update(self):
        cv2.imshow(self.map_win_name, self.draw_map())
        cv2.imshow(self.obs_win_name, self.draw_obs())

    def run(self):
        self.update()
        while self.running:
            c = cv2.waitKey(30)
            if self.on_key_press(c):
                self.update()

