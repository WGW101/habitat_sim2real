import argparse
import threading
import time

import habitat
import habitat_sim2real


DEFAULT_ARGS = {"config_path": "configs/locobot_pointnav_citi_sim.yaml",
                "freq": 20}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-path", "-c")
    parser.add_argument("--freq", "-f", type=float)
    parser.add_argument("extra_args", nargs=argparse.REMAINDER)
    parser.set_defaults(**DEFAULT_ARGS)
    return parser.parse_args()


class UpdateMetricsThread(threading.Thread):
    def __init__(self, env, freq=30):
        self.env = env
        self.period = 1 / freq
        self.running = True

    def run(self):
        while self.running:
            self.env.task.measurements.update_measures(episode=self.env.current_episode,
                                                       task=self.env.task)
            time.sleep(self.period)


def main(args):
    cfg = habitat_sim2real.get_config(args.config_path)
    with habitat.Env(cfg) as env:
        for _ in env.episodes:
            env.reset()
            t = UpdateMetricsThread(env, args.freq)
            t.start()
            env.sim.intf_node.move_to_absolute(env.current_episode.goals[0].position)
            t.running = False
            t.join()
            m = env.get_metrics()
            print(m)

if __name__ == "__main__":
    main(parse_args())
