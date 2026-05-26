import jax
import os
import functools

os.environ["QT_QPA_PLATFORM"] = "xcb"
# os.environ["QT_QPA_PLATFORM"] = "wayland"        # must be before importing matplotlib
# os.environ["QT_LOGGING_RULES"] = "*.warning=false"  # suppress runtime warnings
import matplotlib

matplotlib.use("TkAgg")  # Tk
# matplotlib.use("QtAgg")   # Qt
# matplotlib.use("Agg")   # Qt
import matplotlib.pyplot as plt
from jax import numpy as jnp
from gaussians import Gaussian, plot_gaussian
from gp import ParametricGaussianProcess, GP_Plot
from random_data import sample_noisy_linear
from shared import cmap_wg, cmap_wr, rgb


jax.config.update("jax_enable_x64", True)


prior = Gaussian(mu=jnp.zeros(2), Sigma=jnp.eye(2))

phi = lambda x: jnp.hstack([jnp.ones_like(x), x])

sigma = 0.4
key = jax.random.PRNGKey(0)
Xn, Yn = sample_noisy_linear(key, n=30, a=1.5, b=-0.5, sigma=0.3)


def gaussian_features(
    x: jnp.ndarray, num_features: float, cmin: float, cmax: float, ell=1.0
) -> jnp.ndarray:
    """Feature map for a Gaussian basis function"""
    # Output shape: (n_samples, order)
    return (jnp.sqrt(cmax - cmin)) * jnp.exp(
        -(((x - jnp.linspace(cmin, cmax, num_features)) / ell) ** 2)
    )


fig, ax = plt.subplots(1)
x = jnp.linspace(-8, 8, 300)[:, None]
ax.plot(
    x, gaussian_features(x, num_features=20, cmin=-8, cmax=8), lw=1, color=rgb.tue_red
)
ax.set_xlabel("$x$")
ax.set_ylabel(r"$\phi(x)")

# plt.show()


num_features = 10

prior_w = Gaussian(mu=jnp.zeros(num_features), Sigma=jnp.eye(num_features))

import ipdb; ipdb.set_trace()


phi = functools.partial(
    gaussian_features, num_features=num_features, cmin=-8, cmax=8, ell=1.0
)

gp = ParametricGaussianProcess(phi=phi, prior=prior_w)

fig, ax = plt.subplots(1)
x = jnp.linspace(-8, 8, 300)[:, None]
yrange = (-5, 5)

GP_Plot(
    gp,
    ax,
    yrange=yrange,
    yres=1000,
    color=rgb.tue_red,
    mean_kwargs={"label": "GP mean"},
    std_kwargs={"alpha": 0.5, "label": "GP std", "cmap": cmap_wr},
)

ax.plot(
    x,
    gaussian_features(x, num_features=num_features, cmin=-8, cmax=8),
    lw=0.75,
    color=rgb.tue_dark,
    alpha=0.5,
)
ax.set_ylim(*yrange)

plt.show()

# sigma = 1.5
# posterior = prior.condition(phi(Xn[:, None]), Yn, sigma**2 * jnp.eye(len(Xn)))

# x = jnp.linspace(-5, 5, 100)[:, None]
# f_prior = phi(x) @ prior
# f_posterior = phi(x) @ posterior
