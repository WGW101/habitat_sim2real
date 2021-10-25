import numpy as np
import quaternion
import cv2

import habitat
import habitat_sim2real


cfg = habitat_sim2real.get_config().SIMULATOR
cfg.defrost()
cfg.TYPE = "Sim-v1"
cfg.AGENT_0.SENSORS.append("SCAN_SENSOR")
cfg.SCAN_SENSOR.POINTS_FORMAT = "POLAR"
cfg.SCAN_SENSOR.ORIENTATION = [0.167 * np.pi, np.pi, 0]
cfg.SCAN_SENSOR.MIN_ANGLE = -0.5 * np.pi
cfg.SCAN_SENSOR.MAX_ANGLE = 0.5 * np.pi
cfg.freeze()

SHOW_SCANLINES = True

with habitat.sims.make_sim(cfg.TYPE, config=cfg) as sim:
    obs = sim.reset()
    
    h = sim.get_agent_state().position[1]
    navmask = sim.pathfinder.get_topdown_view(0.01, h + 0.5)
    origin, _ = sim.pathfinder.get_bounds()
    origin[1] = h

    scan = sim.sensor_suite.sensors["scan"]

    while True:
        img = np.full(navmask.shape + (3,), 255, dtype=np.uint8)
        img[navmask] = 127

        s = sim.get_agent_state()
        j0, _, i0 = ((s.position - origin) / 0.01).astype(np.int64)
        cv2.circle(img, (j0, i0), 20, (255, 0, 0), -1)
        head = (s.rotation * np.quaternion(0, 0, 0, -0.2) * s.rotation.conj()).vec + s.position
        j1, _, i1 = ((head - origin) / 0.01).astype(np.int64)
        cv2.line(img, (j0, i0), (j1, i1), (0, 0, 0), 3)

        z = obs["scan"]
        s = scan.get_state(s)
        rel = np.zeros((z.shape[0], 4))
        if cfg.SCAN_SENSOR.POINTS_FORMAT == "POLAR":
            rel[:, 1] = -z[:, 0] * np.sin(z[:, 1])
            rel[:, 3] = -z[:, 0] * np.cos(z[:, 1])
        else: # CARTESIAN
            rel[:, 1] = -z[:, 1]
            rel[:, 3] = -z[:, 0]
        q = quaternion.from_float_array(rel)
        pts = quaternion.as_float_array(s.rotation * q * s.rotation.conj())[:, 1:] + s.position
        j, _, i =  ((pts - origin) / 0.01).astype(np.int64).T
        if SHOW_SCANLINES:
            for ii, jj in zip(i, j):
                cv2.line(img, (j0, i0), (jj, ii), (0, 0, 255))
        else:
            mask = (0 <= i) & (i < navmask.shape[0]) & (0 <= j) & (j < navmask.shape[1])
            img[i[mask], j[mask]] = (0, 0, 255)

        cv2.imshow("Map", img)
        c = cv2.waitKey()
        if c == ord('w'):
            obs = sim.step(1)
        elif c == ord('a'):
            obs = sim.step(2)
        elif c == ord('d'):
            obs = sim.step(3)
        else:
            break
