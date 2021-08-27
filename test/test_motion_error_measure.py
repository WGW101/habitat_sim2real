import pytest
import habitat
import random


EPSILON = 1e-2


def test_motion_error_measure():
    from habitat_sim2real.tasks.real.motion_error_measure import MotionErrorMeasure, MotionErrors

    assert habitat.registry.get_measure("MotionErrorMeasure") is MotionErrorMeasure

    cfg = habitat.get_config()
    cfg.defrost()
    cfg.SIMULATOR.HABITAT_SIM_V0.ALLOW_SLIDING = False
    cfg.TASK.MEASUREMENTS.append("MOTION_ERROR")
    cfg.TASK.MOTION_ERROR = habitat.Config()
    cfg.TASK.MOTION_ERROR.TYPE = "MotionErrorMeasure"
    cfg.freeze()
    with habitat.Env(cfg) as env:
        env.reset()
        m = env.get_metrics()
        assert "motion_error" in m
        err = m["motion_error"]
        assert isinstance(err, dict)
        assert all(err_dim in err for err_dim in MotionErrors._fields)
        assert err["action"] == "START"
        assert not err["collision"]
        assert all(err[err_dim] == 0 for err_dim in ("longitudinal", "lateral", "angular"))

        for _ in range(100):
            a = random.choice(("MOVE_FORWARD", "TURN_LEFT", "TURN_RIGHT"))
            env.step(a)
            err = env.get_metrics()["motion_error"]
            print(err)
            if err["collision"]:
                assert abs(err["longitudinal"]) > EPSILON
            else:
                assert abs(err["longitudinal"]) < EPSILON
                assert abs(err["lateral"]) < EPSILON
            assert abs(err["angular"]) < EPSILON
