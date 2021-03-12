import os
os.environ["GLOG_minloglevel"] = "2"
os.environ["MAGNUM_LOG"] = "quiet"
import logging

import numpy as np
from quaternion import rotate_vectors as quat_rot
import cv2

import habitat
habitat.logger.setLevel(logging.ERROR)
from habitat.utils.visualizations.maps import get_topdown_map_from_sim, colorize_topdown_map
from habitat_sim import NavMeshSettings

from habitat_sim2real import BaseSimulatorViewer


CFG_PATH = "configs/locobot_pointnav_gibson.yaml"
SCENE_PATH = "data/scene_datasets/nle/09c52b4f283344bf92e4a71df646c9b4.glb"
NAVMESH_SETTINGS_BOUNDS = {"agent_max_slope": (5, 60), "agent_max_climb": (0.0, 0.50)}


class NavmeshInspector(BaseSimulatorViewer):
    def __init__(self, sim_cfg, win_basename="Inspector", scale=None):
        super().__init__(sim_cfg, win_basename="Inspector", scale=None)
        self.sim.navmesh_visualization = True
        self.navmesh_settings = NavMeshSettings()
        self.navmesh_settings.set_defaults()
        self.navmesh_settings.agent_radius = sim_cfg.AGENT_0.RADIUS
        self.navmesh_settings.agent_height = sim_cfg.AGENT_0.HEIGHT

    def on_key_press(self, key_code):
        super_update = super().on_key_press(key_code)
        update = True
        if key_code == ord('u'):
            self.update_navmesh("agent_max_climb", +0.01)
        elif key_code == ord('j'):
            self.update_navmesh("agent_max_climb", -0.01)
        elif key_code == ord('i'):
            self.update_navmesh("agent_max_slope", +5)
        elif key_code == ord('k'):
            self.update_navmesh("agent_max_slope", -5)
        elif key_code == ord('o'):
            self.save_navmesh()
        else:
            update = False
        return update or super_update

    def update_navmesh(self, attr, mod):
        prv_val = getattr(self.navmesh_settings, attr)
        val_min, val_max = NAVMESH_SETTINGS_BOUNDS[attr]
        val = max(val_min, min(prv_val + mod, val_max))
        if val != prv_val:
            setattr(self.navmesh_settings, attr, val)
            success = self.sim.recompute_navmesh(self.sim.pathfinder, self.navmesh_settings)
            if success:
                self.obs = self.sim.get_observations_at()
                self.map_img = colorize_topdown_map(get_topdown_map_from_sim(self.sim))
                if self.scale:
                    self.map_img = cv2.resize(self.map_img, None, fx=self.scale, fy=self.scale)

    def save_navmesh(self):
        out = self.cfg.SCENE.replace(".glb", ".navmesh")
        self.sim.pathfinder.save_nav_mesh(out)


if __name__ == "__main__":
    sim_cfg = habitat.get_config(CFG_PATH).SIMULATOR
    sim_cfg.defrost()
    sim_cfg.SCENE = SCENE_PATH
    sim_cfg.freeze()

    inspect = NavmeshInspector(sim_cfg, scale=0.6)
    inspect.run()
