import numpy as np
from matplotlib import pyplot as plt
from habitat_sim2real import perlin_1d, perlin_2d
import time


dur = []
for _ in range(100):
    start = time.monotonic()
    noise = perlin_1d(480, 5, 0.0083, 3)
    dur.append(time.monotonic() - start)
    plt.plot(np.arange(480), noise)
print("Mean duration on 100 1d perlin noise = {:.3f}ms".format(1000 * sum(dur) / len(dur)))
    

_, axes = plt.subplots(4, 4)
dur = []
for ax in axes.flatten():
    start = time.monotonic()
    noise = perlin_2d(640, 480, 10, 0.013, 2)
    dur.append(time.monotonic() - start)
    ax.imshow(noise)
print("Mean duration on 16 2d perlin noise = {:.3f}ms".format(1000 * sum(dur) / len(dur)))

plt.show()

