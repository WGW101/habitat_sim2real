import numpy as np


def _my_interp(x, xp, fp, func):
    w = xp[1:] - xp[:-1]
    idx = np.digitize(x, xp) - 1
    xi = xp[idx]
    xf = (x - xi) / w[idx]
    f = func(xf)
    return fp[idx] * (1 - f) + fp[idx + 1] * f

def poly5_interp(x, xp, fp):
    return _my_interp(x, xp, fp, func=lambda xf: 6 * xf**5 - 15 * xf**4 + 10 * xf**3)

def cosine_interp(x, xp, fp):
    return _my_interp(x, xp, fp, func=lambda xf: 0.5 * (1 - np.cos(np.pi * xf)))


def perlin_1d(size, amp, freq, octaves=1, persist=0.5, seed=None, x=None):
    if seed is not None:
        np.random.seed(seed)
    if x is None:
        x = np.arange(size)
    y = np.zeros(size)
    for _ in range(octaves):
        n = int(size * freq)
        rp = (2 * np.random.rand(n) - 1) * amp
        xp = np.linspace(0, size, n)
        y += cosine_interp(x, xp, rp)
        amp *= persist
        freq *= 2
    if octaves > 1:
        y *= (1 - persist) / (1 - persist**octaves)
    return y


def perlin_2d(width, height, amp, freq, octaves=1, persist=0.5, seed=None, x=None, y=None):
    if seed is not None:
        np.random.seed(seed)
    if x is None:
        x = np.arange(width)
    if y is None:
        y = np.arange(height)
    img = np.zeros((height, width))
    for _ in range(octaves):
        n_x = int(width * freq)
        n_y = int(height * freq)
        gp_a = 2 * np.pi * np.random.rand(n_y, n_x)
        gp = np.dstack((np.cos(gp_a), np.sin(gp_a)))

        xp = np.linspace(0, width, n_x)
        w_x = xp[1:] - xp[:-1]
        idx_x = np.digitize(x, xp) - 1
        xf = (x - xp[idx_x]) / w_x[idx_x]
        xs = 6 * xf**5 - 15 * xf**4 + 10 * xf**3

        yp = np.linspace(0, height, n_y)
        w_y = yp[1:] - yp[:-1]
        idx_y = np.digitize(y, yp) - 1
        yf = (y - yp[idx_y]) / w_y[idx_y]
        ys = 6 * yf**5 - 15 * yf**4 + 10 * yf**3

        g_ul = gp[idx_y[:, None], idx_x[None, :]]
        dot_ul = g_ul[:, :, 0] * xf[None, :] + g_ul[:, :, 1] * yf[:, None]
        g_ur = gp[idx_y[:, None], idx_x[None, :] + 1]
        dot_ur = g_ur[:, :, 0] * (1 - xf[None, :]) + g_ur[:, :, 1] * yf[:, None]
        dot_u = (1 - xs[None, :]) * dot_ul + xs[None, :] * dot_ur

        g_bl = gp[idx_y[:, None] + 1, idx_x[None, :]]
        dot_bl = g_bl[:, :, 0] * xf[None, :] + g_bl[:, :, 1] * (1 - yf[:, None])
        g_br = gp[idx_y[:, None] + 1, idx_x[None, :] + 1]
        dot_br = g_br[:, :, 0] * (1 - xf[None, :]) + g_br[:, :, 1] * (1 - yf[:, None])
        dot_b = (1 - xs[None, :]) * dot_bl + xs[None, :] * dot_br

        dot = (1 - ys[:, None]) * dot_u + ys[:, None] * dot_b
        img += amp * dot

        amp *= persist
        freq *= 2

    if octaves > 1:
        img *= (1 - persist) / (1 - persist**octaves)
    return img
