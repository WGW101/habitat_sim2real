import numpy
import quaternion
from habitat_sim import Simulator as HabitatSimulator


def get_observation_from_sim(sim, x, z, orient, y=None):
    """
    Render a RGB-D observation at a given 2d pose in sim.

    :param sim: Simulator pre-loaded with the scene
    :param x: X coodinate of the agent/robot in habitat ref frame
    :param z: Z coodinate of the agent/robot in habitat ref frame
    :param orient: Orientation of the agent/robot base in degrees
    :param y: Y coodinate of the agent/robot in habitat ref frame
        Default to current agent/robot height
    :type sim: habitat.Simulator
    :type x: float
    :type z: float
    :type orient: float
    :type y: float
    :return: A dictionary containing observations "rgb" and "depth"
    :rtype: dict(str: numpy.ndarray)
    """
    if y is None:
        y = sim.get_agent_state().position[1]
    pos = numpy.array([x, y, z])
    a = numpy.radians(orient)
    rot = [0, numpy.sin(0.5 * a), 0, numpy.cos(0.5 * a)]
    return sim.get_observations_at(pos, rot, True)


def reconfigure_rgbd_camera_in_sim(sim, x=None, y=None, z=None,
                                        pitch=None, yaw=None, roll=None,
                                        rgb_to_depth=None,
                                        rgb_hfov=None, depth_hfov=None,
                                        depth_min=None, depth_max=None):
    """
    Change sensor configuration of the agent in sim.

    :param sim: Simulator pre-loaded with the scene
    :param x: Lateral position of the (depth) camera relative to the agent/robot base
    :param y: Vertical position of the (depth) camera relative to the agent/robot base
    :param z: Forward position of the (depth) camera relative to the agent/robot base
    :param pitch: Pitch (X rotation) of the camera relative to the agent/robot base
    :param yaw: Yaw (Y rotation) of the camera relative to the agent/robot base
    :param roll: Roll (Z rotation) of the camera relative to the agent/robot base
    :param rgb_to_depth: Distance between the rgb and depth camera optical centers (baseline)
    :param rgb_hfov: Horizontal field of view of the rgb camera in degrees
    :param depth_hfov: Horizontal field of view of the depth camera in degrees
    :param depth_min: Minimum depth value (clipped)
    :param depth_max: Maximum depth value (clipped)
    :type x: float
    :type y: float
    :type z: float
    :type pitch: float
    :type yaw: float
    :type roll: float
    :type rgb_to_depth: float
    :type rgb_hfov: int
    :type depth_hfov: int
    :type depth_min: float
    :type depth_max: float
    """
    rgb_cfg = sim._sensor_suite.sensors["rgb"].config
    rgb_cfg.defrost()
    depth_cfg = sim._sensor_suite.sensors["depth"].config
    depth_cfg.defrost()

    if rgb_to_depth is None:
        rgb_to_depth = rgb_cfg.POSITION[0] - depth_cfg.POSITION[0]

    if pitch is not None:
        depth_cfg.ORIENTATION[0] = pitch
        rgb_cfg.ORIENTATION[0] = pitch
    if yaw is not None:
        depth_cfg.ORIENTATION[1] = yaw
        rgb_cfg.ORIENTATION[1] = yaw
    if roll is not None:
        depth_cfg.ORIENTATION[2] = roll
        rgb_cfg.ORIENTATION[2] = roll

    rot = quaternion.from_euler_angles(*depth_cfg.ORIENTATION)

    if x is not None:
        depth_cfg.POSITION[0] = x
    if y is not None:
        depth_cfg.POSITION[1] = y
    if z is not None:
        depth_cfg.POSITION[2] = z
    if any(arg is not None for arg in (x, y, z, pitch, yaw, roll, rgb_to_depth)):
        # RGB has to be moved
        rel = rot * quaternion.quaternion(0, rgb_to_depth, 0, 0) * rot.conj()
        rgb_cfg.POSITION[0] = depth_cfg.POSITION[0] + rel.x
        rgb_cfg.POSITION[1] = depth_cfg.POSITION[1] + rel.y
        rgb_cfg.POSITION[2] = depth_cfg.POSITION[2] + rel.z
    if rgb_hfov is not None:
        rgb_cfg.HFOV = rgb_hfov
    if depth_hfov is not None:
        depth_cfg.HFOV = depth_hfov
    if depth_min is not None:
        depth_cfg.MIN_DEPTH = depth_min
    if depth_max is not None:
        depth_cfg.MAX_DEPTH = depth_max

    rgb_cfg.freeze()
    depth_cfg.freeze()
    s = sim.get_agent_state()
    sim.sim_config = sim.create_sim_config(sim._sensor_suite)
    HabitatSimulator.reconfigure(sim, sim.sim_config)
    return sim.get_observations_at(s.position, s.rotation, True)


if __name__ == "__main__":
    import cv2
    import habitat
    cfg = habitat.get_config("configs/locobot_pointnav_citi_sim.yaml").SIMULATOR
    sim = habitat.sims.make_sim(cfg.TYPE, config=cfg)
    obs = sim.reset()

    while True:
        disp = cv2.cvtColor((obs["depth"] * 255).astype(numpy.uint8), cv2.COLOR_GRAY2BGR)
        cv2.imshow("Obs", numpy.vstack((obs["rgb"][:, :, ::-1], disp)))
        cv2.waitKey(1)

        args = input("Cmd? > ").split()
        if args[0] == "obs":
            x, z, orient = (float(s) for s in args[1:])
            obs = get_observation_from_sim(sim, x, z, orient)
        elif args[0] == "cfg":
            params = {}
            for k, v in zip(args[1::2], args[2::2]):
                if k in ("rgb_hfov", "depth_hfov"):
                    params[k] = int(v)
                else:
                    params[k] = float(v)
            obs = reconfigure_rgbd_camera_in_sim(sim, **params)
