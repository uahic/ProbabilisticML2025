from typing import Self

from jax import numpy as jnp
from jax import vmap
from jaxtyping import Array, Float
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colors import Colormap
import matplotlib.pyplot as plt

from gaussian import Gaussian

mu = jnp.array([1.0, 2.0])
Sigma = jnp.array([[1.0, 0.5], [0.5, 2.0]])

G = Gaussian(mu, Sigma)


class rgb:
    tue_dark = (0.1, 0.1, 0.1)  # Beispielwert
    tue_green = (0.1, 0.3, 0.3)


cmp_wd = LinearSegmentedColormap.from_list("wd", ["w", rgb.tue_dark], N=1024)


X, Y = jnp.mgrid[-2:4:200j, -2:4:200j]
nplot = X.shape[0]
XY = jnp.dstack((X, Y)).reshape(nplot**2, 2)
Z = vmap(G.pdf)(XY).reshape(nplot, nplot)


fig, ax = plt.subplots()
ax.contourf(X, Y, Z, levels=200, cmap=cmp_wd)
ax.plot(mu[0], mu[1], "o", color=rgb.tue_green, markersize=3)
plt.plot()
