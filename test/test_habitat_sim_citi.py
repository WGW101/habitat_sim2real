#!/usr/bin/env python3

import habitat_sim
import cv2
import numpy

sim_cfg = habitat_sim.SimulatorConfiguration()
sim_cfg.scene.id = "./data/citi/4d6ab9cc04f24c87aed802698f957b7a.glb"
sim_cfg.allow_sliding = False

ag_cfg = habitat_sim.agent.AgentConfiguration()
color_spec = habitat_sim.SensorSpec()
color_spec.uuid = "color_sensor"
color_spec.sensor_type = habitat_sim.SensorType.COLOR
color_spec.resolution = [480, 640]
color_spec.position = 0.047 * habitat_sim.geo.FRONT \
                    + 0.015 * habitat_sim.geo.LEFT \
                    + 0.589 * habitat_sim.geo.UP
color_spec.parameters["hfov"] = "69"
depth_spec = habitat_sim.SensorSpec()
depth_spec.uuid = "depth_sensor"
depth_spec.sensor_type = habitat_sim.SensorType.DEPTH
depth_spec.resolution = [480, 640]
depth_spec.position = 0.047 * habitat_sim.geo.FRONT \
                    + 0.000 * habitat_sim.geo.LEFT \
                    + 0.589 * habitat_sim.geo.UP
depth_spec.parameters["hfov"] = "90"
ag_cfg.sensor_specifications = [color_spec, depth_spec]
ag_cfg.action_space = {k: habitat_sim.agent.ActionSpec(k, habitat_sim.agent.ActuationSpec(amount=v))
                       for k,v in (("move_forward",  0.25),
                                   ("turn_left",    10.0),
                                   ("turn_right",   10.0),
                                   ("look_up",      15.0),
                                   ("look_down",    15.0))}

cfg = habitat_sim.Configuration(sim_cfg, [ag_cfg])
sim = habitat_sim.Simulator(cfg)

obs = sim.reset()
while True:
    cv2.imshow("Color", obs["color_sensor"][:, :, ::-1])
    cv2.imshow("Depth", obs["depth_sensor"] / 5)
    c = cv2.waitKey()
    if c == ord("w"):
        obs = sim.step("move_forward")
    elif c == ord("a"):
        obs = sim.step("turn_left")
    elif c == ord("d"):
        obs = sim.step("turn_right")
    elif c == ord("r"):
        obs = sim.step("look_up")
    elif c == ord("f"):
        obs = sim.step("look_down")
    else:
        break
cv2.destroyAllWindows()
