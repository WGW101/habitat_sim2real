import numpy as np
import cv2

import habitat
from habitat.sims.habitat_simulator.habitat_simulator import HabitatSimDepthSensor

from habitat_sim2real import perlin_1d, perlin_2d

@habitat.registry.register_sensor
class RealisticHabitatSimDepthSensor(HabitatSimDepthSensor):
    def __init__(self, config):
        super().__init__(config)
        self.grad_thresh = 0.3
        self.occl_band_noise_params ={"size": config.HEIGHT,
                                      "amp": 8,
                                      "freq": 0.04,
                                      "octaves": 3}
        self.img_noise_params = {"width": config.WIDTH,
                                 "height": config.HEIGHT,
                                 "amp": 0.6,
                                 "freq": 0.016,
                                 "octaves": 4}
        self.speckles_thresh = 0.84

    def get_observation(self, sim_obs):
        obs = sim_obs.get(self.uuid, None) # Raw obs, depth in meters
        assert obs is not None

        grad_x = cv2.Sobel(obs, cv2.CV_32F, 1, 0, ksize=cv2.FILTER_SCHARR) / 32
        _, mask = cv2.threshold(grad_x, -self.grad_thresh, 1, cv2.THRESH_BINARY_INV)

        band_noise = perlin_1d(**self.occl_band_noise_params) \
                + 2 * self.occl_band_noise_params["amp"]
        for y, x in zip(*np.nonzero(mask)):
            b = band_noise[y]
            if obs[y, x] > 0:
                b = b / obs[y, x]**0.5
            obs[y, int(x - b):x] = np.nan

        band_noise = perlin_1d(**self.occl_band_noise_params) \
                + 2 * self.occl_band_noise_params["amp"]
        for y in range(480):
            b = band_noise[y]
            if obs[y, 0] > 0:
                b = 2 * b / obs[y, 0]**0.5
            obs[y, :int(b)] = np.nan

        img_noise = perlin_2d(**self.img_noise_params) + 1
        obs *= img_noise

        obs[img_noise < self.speckles_thresh] = np.nan
        
        obs = np.clip(obs, self.config.MIN_DEPTH, self.config.MAX_DEPTH)[:, :, None]
        if self.config.NORMALIZE_DEPTH:
            obs = (obs - self.config.MIN_DEPTH) / (self.config.MAX_DEPTH - self.config.MIN_DEPTH)
        return obs
