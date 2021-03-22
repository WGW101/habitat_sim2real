import numpy as np
from quaternion import quaternion as quat
import cv2

import habitat
from habitat.utils.visualizations.maps import get_topdown_map_from_sim, colorize_topdown_map


class BaseSimulatorViewer:
    def __init__(self, sim_cfg, win_basename="Simulator", scale=None):
        self.cfg = sim_cfg
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
        self.map_height = self.sim.get_agent_state().position[1]

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
            self.teleport_agent_on_map(self.drag_start, self.drag_vec)
            self.drag_start = None
            self.drag_vec = None
            self.update()

    def on_key_press(self, key_code):
        update = True
        if key_code == ord('x'):
            self.running = False
        elif key_code == ord('r'):
            self.obs = self.sim.reset()
        elif key_code == ord('w'):
            self.move_agent(self.cfg.FORWARD_STEP_SIZE, 0)
        elif key_code == ord('s'):
            self.move_agent(-self.cfg.FORWARD_STEP_SIZE, 0)
        elif key_code == ord('a'):
            self.move_agent(0, -self.cfg.FORWARD_STEP_SIZE)
        elif key_code == ord('d'):
            self.move_agent(0, self.cfg.FORWARD_STEP_SIZE)
        elif key_code == ord('q'):
            self.obs = self.sim.step(2)
        elif key_code == ord('e'):
            self.obs = self.sim.step(3)
        else:
            update = False
        return update

    def teleport_agent_on_map(self, pos, head=None):
        pos = self.project_map_to_pos(self.drag_start)
        if head is None:
            rot = [0.0, 1.0, 0.0, 0.0]
        else:
            yaw = np.pi + np.arctan2(head[0], head[1])
            rot = [0, np.sin(0.5 * yaw), 0, np.cos(0.5 * yaw)]
        self.obs = self.sim.get_observations_at(pos, rot, True)

    def move_agent(self, forward=0, right=0):
        if forward == 0 and right == 0:
            return
        s = self.sim.get_agent_state()
        q = quat(0, right, 0, -forward)
        move = s.rotation * q * s.rotation.conjugate()
        pos = s.position + move.vec
        self.obs = self.sim.get_observations_at(pos, s.rotation, True)

    def project_pos_to_map(self, pos):
        return ((pos[::2] - self.map_origin) / self.map_resolution).astype(np.int64)

    def project_map_to_pos(self, uv):
        xz = uv * self.map_resolution + self.map_origin
        return np.array([xz[0], self.map_height, xz[1]])

    def draw_agent_on_map(self, disp, pos=None, head=None, color=(255, 0, 0)):
        if pos is None:
            s = self.sim.get_agent_state()
            pos = self.project_pos_to_map(s.position)
            q = quat(0, 0, 0, -1)
            head = (s.rotation * q * s.rotation.conjugate()).vec[::2]
        elif head is None:
            head = np.array([0.0, 1.0])
        cv2.circle(disp, tuple(pos), 5, color, -1)
        end = (pos + 10 * head).astype(np.int64)
        cv2.line(disp, tuple(pos), tuple(end), color, 3)

    def draw_map(self):
        disp = self.map_img.copy()
        self.draw_agent_on_map(disp)
        if self.drag_vec is not None:
            self.draw_agent_on_map(disp, self.drag_start, self.drag_vec, (0, 255, 0))
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
        self.running = True
        self.update()
        while self.running:
            c = cv2.waitKey(30)
            if self.on_key_press(c):
                self.update()

