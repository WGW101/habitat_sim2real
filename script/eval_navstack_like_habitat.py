import argparse
import threading
import time

import numpy as np

import habitat
import habitat_sim2real


DEFAULT_ARGS = {"config_path": "configs/locobot_pointnav_citi_sim.yaml",
                "freq": 20,
                "out_file": "out/eval_navstack_like_habitat.log"}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-path", "-c")
    parser.add_argument("--freq", "-f", type=float)
    parser.add_argument("--out-file", "-o")
    parser.add_argument("extra_args", nargs=argparse.REMAINDER)
    parser.set_defaults(**DEFAULT_ARGS)
    return parser.parse_args()


class UpdateMetricsThread(threading.Thread):
    def __init__(self, env, freq):
        self.env = env
        self.period = 1 / freq
        self.running = True
        self.last_pos = None
        self.cumul_dist = None
        self.optim_dist = None
        super().__init__()

    def run(self):
        self.last_pos = self.env.sim.get_agent_state().position
        self.cumul_dist = 0
        self.optim_dist = self.env.get_metrics()['distance_to_goal']
        while self.running:
            pos = self.env.sim.get_agent_state().position
            if not np.allclose(pos, self.last_pos):
                self.cumul_dist += np.linalg.norm(pos - self.last_pos)
                self.last_pos = pos
            time.sleep(self.period)

    def get_metrics(self):
        d = self.env.sim.geodesic_distance(self.env.sim.get_agent_state().position,
                                           self.env.current_episode.goals[0].position)
        s = 1 if d < self.env.task._config.SUCCESS.SUCCESS_DISTANCE else 0
        p = max(0, 1 - d / self.optim_dist)
        return {"distance_to_goal": d,
                "success": s,
                "spl": s * self.optim_dist / max(self.optim_dist, self.cumul_dist),
                "soft_spl": p * self.optim_dist / max(self.optim_dist, self.cumul_dist)}


def main(args):
    cfg = habitat_sim2real.get_config(args.config_path)
    with open(args.out_file, 'wt') as logf:
        with habitat.Env(cfg) as env:
            for _ in env.episodes[:-1]:
                next(env._episode_iterator)
            for _ in env.episodes:
                input("ENTER to start next episode> ")
                env.reset()
                print(f"Episode: {env.current_episode.episode_id}")
                print(f"Episode: {env.current_episode.episode_id}", file=logf)
                t = UpdateMetricsThread(env, args.freq)
                t.start()
                env.sim.intf_node.cancel_move_on_bump = False
                env.sim.intf_node.move_to_absolute(env.current_episode.goals[0].position)
                t.running = False
                t.join()
                print(t.get_metrics())
                print(t.get_metrics(), file=logf)


if __name__ == "__main__":
    main(parse_args())
