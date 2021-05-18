from habitat_sim2real import ROSRobot, get_config

cfg = get_config("configs/locobot_pointnav_citi_sim.yaml").SIMULATOR
cfg.defrost()
cfg.TYPE = "ROS-Robot-v0"
cfg.freeze()

print("Initializing ROSRobot...", end=' ', flush=True)
sim = ROSRobot(cfg)
print("Done!")
print("Sensor suite:", sim.sensor_suite)
print("Action space:", sim.action_space)
print("Up vector:", sim.up_vector)
print("Forward vector:", sim.forward_vector)

cfg2 = cfg.clone()
cfg2.defrost()
cfg2.RGB_SENSOR.ORIENTATION[0] = -0.3
cfg2.freeze()
print("Reconfiguring with cam tilted...", end=' ', flush=True)
sim.reconfigure(cfg2)
print("Done!")

print("Resetting...", end=' ', flush=True)
obs = sim.reset()
print("Done!")
print("Please check that camera is tilted up!")
input("Press ENTER to reconfigure and reset with defaults")

sim.reconfigure(cfg)
obs = sim.reset()
print("Done!")
print("Received observations with keys:", obs.keys())

print("Getting observations at current pose...", end=' ', flush=True)
obs2 = sim.get_observations_at()
print("Done!")
print("Received observations with keys:", obs2.keys())

print("Trying to get observations at diff pose...")
try:
    obs2 = sim.get_observations_at([1, 0, 2])
    error = None
except RuntimeError as e:
    error = e
print("Done!")
print("Raised error:", error)

print("Getting agent state...", end=' ', flush=True)
s = sim.get_agent_state()
print("Done!")
print("Received state:", s)

print("Seeding RNG...", end=' ', flush=True)
sim.seed(123456)
print("Done!")

print("Sampling navigable point...", end=' ', flush=True)
pt = sim.sample_navigable_point()
print("Done!")
print("Received point:", pt)

print("Computing geodesic distance...", end=' ', flush=True)
d = sim.geodesic_distance(s.position, pt)
print("Done!")
print("Received distance:", d)

print("Getting map...", end=' ', flush=True)
topdown_map = sim.get_topdown_map()
print("Done!")
print("Received map of size:", topdown_map.shape)

print("WARNING! Robot is going to move!")
input("Press ENTER to move forward")
obs = sim.step(1)
print("Done!")
print("Previous state:", s)
s = sim.get_agent_state()
print("New state:", s)

input("Press ENTER to turn left")
obs = sim.step(2)
print("Done!")
print("Previous state:", s)
s = sim.get_agent_state()
print("New state:", s)

input("Press ENTER to turn right")
obs = sim.step(3)
print("Done!")
print("Previous state:", s)
s = sim.get_agent_state()
print("New state:", s)

input("Press ENTER to look up")
obs = sim.step(4)
print("Done!")

input("Press ENTER to look down")
obs = sim.step(5)
print("Done!")
