import os

os.environ["QT_QPA_PLATFORM"] = "xcb"
import matplotlib

matplotlib.use("TkAgg")  # Tk
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import pandas as pd
from jax import jit
from jax import numpy as jnp
from jax import random
from probML2025.shared import rgb, cmap_wr, cmap_wd, cmap_bwo, KEY
from kalman import KalmanFilter

df = pd.read_csv("trajectory_data.csv")
x = jnp.array(df["X"])
xdot = jnp.array(df["dX"])
y = jnp.array(df["Y"])
ydot = jnp.array(df["dY"])
n = len(x)
dt = df["t"][1] - df["t"][0]

key = random.PRNGKey(0)
sigma = 0.01
measurements = jnp.stack([x, y], axis=1) + sigma * random.normal(key, (len(x), 2))
measurements = jnp.where((jnp.arange(n) % 10 == 0)[:, None], measurements, jnp.nan)


A = jnp.eye(2)
Q = 1e-2 * jnp.eye(2)
H = jnp.eye(2)
R = sigma**2 * jnp.eye(2)
m0 = jnp.array([0.0, 0.0])
P0 = 1e0 * jnp.eye(2)


predict, update, smooth = KalmanFilter(A, Q, H, R)

m = m0
P = P0

means = []
covariances = []
for t in range(n):
    # p(xt | Y:t-1)
    m_, P_ = predict(m, P)

    # in case there is no data at time t
    if not jnp.isnan(measurements[t]).all():
        # p(xt | yt)
        m, P = update(m_, P_, measurements[t])
    else:
        m, P = m_, P_

    means.append(m)
    covariances.append(P)

means = jnp.stack(means)
covariances = jnp.stack(covariances)


def plot_ellipse(ax, m, P, color, alpha=0.2):
    e, V = jnp.linalg.eig(P)
    angle = jnp.arctan2(V[1, 0], V[0, 0]).real
    e = jnp.sqrt(e).real
    center = tuple(m.tolist())
    width = float(2 * e[0])
    height = float(2 * e[1])
    angle_deg = float(angle * 180.0 / jnp.pi)

    ellipse = patches.Ellipse(
        center, width, height, angle=angle_deg, alpha=alpha, color=color
    )
    ax.add_patch(ellipse)


ms = means[-1]
Ps = covariances[-1]
smoothed_means = [ms]
smoothed_covariances = [Ps]
for t in range(n - 2, -1, -1):
    m, P = means[t], covariances[t]
    ms, Ps = smooth(ms, Ps, m, P)
    smoothed_means.append(ms)
    smoothed_covariances.append(Ps)
smoothed_means = jnp.stack(smoothed_means[::-1])
smoothed_covariances = jnp.stack(smoothed_covariances[::-1])


fig, axs = plt.subplots(2, 1)
ax = axs[0]
ax.plot(x, y, label="true_trajectory", color=rgb.tue_red)
ax.plot(means[:, 0], means[:, 1], ".-", color=rgb.tue_orange, label="$m(t)$")
for m, P in zip(means, covariances):
    plot_ellipse(ax, m, P, rgb.tue_dark, alpha=0.1)

ax = axs[1]
# ax.plot(smoothed_means[:,0], smoothed_means[:,1], ".-", label="smoothed_position", color=rgb.tue_dark )


plt.show()
# for m, P in zip(smoothed_means, smoothed_covariances):
#     plot_ellipse(ax,m,P, rgb.tue_dark, alpha=0.1)

# for ax in axs:    
#     ax.plot(
#         measurements[:,0],
#         measurements[:,1],
#         ".",
#         label="measurements",
#         color=rgb.tue_blue
#     )



