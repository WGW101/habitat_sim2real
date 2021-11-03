import numpy as np
import quaternion
import cv2

import habitat
import habitat_sim2real


cfg = habitat_sim2real.get_config().SIMULATOR
cfg.defrost()
cfg.TYPE = "Sim-v1"
cfg.SCENE = "data/scene_datasets/mp3d/5LpN3gDmAk7/5LpN3gDmAk7.glb"
cfg.AGENT_0.SENSORS.append("SCAN_SENSOR")
cfg.SCAN_SENSOR.POINTS_FORMAT = "POLAR"
cfg.SCAN_SENSOR.ORIENTATION = [0, 0, np.pi / 12]
cfg.SCAN_SENSOR.MIN_ANGLE = -np.pi
cfg.SCAN_SENSOR.MAX_ANGLE = np.pi
cfg.freeze()

SHOW_SCANLINES = False

with habitat.sims.make_sim(cfg.TYPE, config=cfg) as sim:
    obs = sim.reset()
    
    h = sim.get_agent_state().position[1]
    navmask = sim.pathfinder.get_topdown_view(0.01, h + 0.5)
    origin, _ = sim.pathfinder.get_bounds()
    origin[1] = h

    rgb_c = 0.5 * np.array([cfg.RGB_SENSOR.WIDTH, cfg.RGB_SENSOR.HEIGHT])
    rgb_f = rgb_c[0] / np.tan(0.5 * np.deg2rad(cfg.RGB_SENSOR.HFOV)) * np.array([-1, 1])
    draw_blk = np.array([[di, dj] for di in range(-1, 2) for dj in range(-1, 2)])

    scan = sim.sensor_suite.sensors["scan"]

    while True:
        img = np.full(navmask.shape + (3,), 255, dtype=np.uint8)
        img[navmask] = 127

        ag_s = sim.get_agent_state()
        j0, _, i0 = ((ag_s.position - origin) / 0.01).astype(np.int64)
        cv2.circle(img, (j0, i0), 20, (255, 0, 0), -1)
        head = (ag_s.rotation * np.quaternion(0, 0, 0, -0.2) * ag_s.rotation.conj()).vec \
                + ag_s.position
        j1, _, i1 = ((head - origin) / 0.01).astype(np.int64)
        cv2.line(img, (j0, i0), (j1, i1), (0, 0, 0), 3)

        z = obs["scan"]
        s = scan.get_state(ag_s)
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
        mask = (0 <= i) & (i < navmask.shape[0]) & (0 <= j) & (j < navmask.shape[1])
        if SHOW_SCANLINES:
            for ii, jj in zip(i[mask], j[mask]):
                cv2.line(img, (j0, i0), (jj, ii), (0, 0, 255))
        else:
            img[i[mask], j[mask]] = (0, 0, 255)

        bgr = obs["rgb"][..., ::-1]
        s = ag_s.sensor_states["rgb"]
        q = np.zeros((z.shape[0], 4))
        q[:, 1:] = pts - s.position
        q = quaternion.from_float_array(q)
        rel = quaternion.as_float_array(s.rotation.conj() * q * s.rotation)[:, 1:]
        rel = rel[rel[:, 2] < 0, :]
        ji = (rgb_f *  rel[:, :2] / rel[:, 2:] + rgb_c).astype(np.int64)
        j, i = (ji[:, None, :] + draw_blk).reshape(-1, 2).T
        mask = (0 <= i) & (i < bgr.shape[0]) & (0 <= j) & (j < bgr.shape[1])
        bgr[i[mask], j[mask]] = (0, 0, 255)

        cv2.imshow("Map", img)
        cv2.imshow("Color", bgr)

        c = cv2.waitKey()
        if c == ord('w'):
            obs = sim.step(1)
        elif c == ord('a'):
            obs = sim.step(2)
        elif c == ord('d'):
            obs = sim.step(3)
        else:
            break

