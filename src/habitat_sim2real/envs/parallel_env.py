from habitat import VectorEnv
from habitat_baselines.common.baseline_registry import baseline_registry


def make_env(cfg):
    env_cls = baseline_registry.get_env(cfg.ENV_NAME)
    dataset = habitat.make_dataset(cfg.TASK_CONFIG.DATASET.TYPE, config=cfg.TASK_CONFIG.DATASET)
    return env_cls(cfg.TASK_CONFIG, dataset)


def make_parallel_envs(cfg):
    cfg.defrost()
    for ag_id in cfg.TASK_CONFIG.SIMULATOR.AGENTS:
        getattr(cfg.TASK_CONFIG.SIMULATOR, ag_id).SENSORS = cfg.SENSORS.copy()
    cfg.freeze()

    scenes = cfg.TASK_CONFIG.DATASET.CONTENT_SCENES
    if "*" in scenes:
        dataset = habitat.make_dataset(cfg.TASK_CONFIG.DATASET.TYPE)
        scenes = dataset.get_scenes_to_load(cfg.TASK_CONFIG.DATASET)

    configs = []
    for proc_idx in range(cfg.NUM_PROCESSES):
        proc_cfg = cfg.clone()
        proc_cfg.defrost()
        proc_cfg.TASK_CONFIG.SEED = cfg.TASK_CONFIG.SEED + proc_idx
        proc_cfg.TASK_CONFIG.DATASET.CONTENT_SCENES = []
        proc_cfg.freeze()
        configs.append(proc_cfg)

    return VectorEnv(make_env, configs)


# Current implem in habitat_baselines:
# -> requires #scenes >= #processes
# -> requires #processes >= #minibatch
# -> round-robin assignment scene->process
# -> minibatch sampling: ???
