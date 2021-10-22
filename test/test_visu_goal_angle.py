import random
import numpy as np
from matplotlib import pyplot as plt
import habitat


def main():
    cfg = habitat.get_config("configs/locobot_pointnav_citi_sim.yaml")
    hfov = np.radians(cfg.SIMULATOR.RGB_SENSOR.HFOV // 2)
    fov_range = np.arange(-hfov, hfov, 0.1)
    d_view = np.full_like(fov_range, 3)

    plt.ion()
    fig, axes = plt.subplots(1, 3, subplot_kw={"projection": "polar"})
    for ax in axes:
        ax.set_theta_zero_location("N")
        ax.set_thetalim(-np.pi, np.pi)
        ax.set_xticks(np.linspace(-np.pi, np.pi, 8, endpoint=False))
        ax.fill_between(fov_range, 0, d_view, color=[(0.2, 0.2, 0.6)])

    axes[0].set_title("Direction through time")
    axes[1].set_title("Live pointgoal (point)")
    axes[2].set_title("Live pointgoal (arrow)")

    with habitat.Env(cfg) as env:
        obs = env.reset()
        _, prv_a = obs["pointgoal_with_gps_compass"]
        prv_a = np.degrees(prv_a)
        prv_draw = []
        for t in range(500):
            obs = env.step(random.randrange(1, 4))
            r, a = obs["pointgoal_with_gps_compass"]
            axes[0].plot([prv_a, a], [t, t+1], c='r')
            for draw in prv_draw:
                draw.remove()
            prv_draw = []
            prv_draw.append(
                axes[1].scatter([a], [np.clip(r, None, 3.5)], c='r')
            )
            prv_draw.append(
                axes[2].plot([a, a], [0, np.clip(r, 0.5, 3.5)], c='r', linewidth=2)[0]
            )
            if r > 3.5:
                prv_draw.append(
                    axes[1].scatter([a, a], [3.7, 3.9], c=[(1, 0.2, 0.2), (1, 0.4, 0.4)])
                )
                prv_draw.append(
                    axes[2].plot([a, a], [3.5, 4.2], c='r', linewidth=2, linestyle=':')[0]
                )
            plt.pause(0.01)
            prv_a = a


if __name__ == "__main__":
    main()
