import habitat_sim
from habitat_sim.agent import ActionSpec, ActuationSpec

sim_cfg = habitat_sim.SimulatorConfiguration()
sim_cfg.scene.id = "data/test/4d6ab9cc04f24c87aed802698f957b7a.glb"
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
ag_cfg.radius = 0.18
ag_cfg.height = 0.65
ag_cfg.sensor_specifications = [color_spec]
ag_cfg.action_space = {k: ActionSpec(k, ActuationSpec(amount=v))
                       for k,v in (("move_forward",  0.25),
                                   ("turn_left",    10.0),
                                   ("turn_right",   10.0),
                                   ("look_up",      15.0),
                                   ("look_down",    15.0))}

cfg = habitat_sim.Configuration(sim_cfg, [ag_cfg])
sim = habitat_sim.Simulator(cfg)

navmesh_settings = habitat_sim.NavMeshSettings()
navmesh_settings.set_defaults()

sim.recompute_navmesh(sim.pathfinder, navmesh_settings, include_static_objects=False)

if sim.pathfinder.is_loaded:
    sim.pathfinder.save_nav_mesh(sim_cfg.scene.id.replace(".glb", ".navmesh"))
