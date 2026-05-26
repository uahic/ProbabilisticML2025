import jax
import os

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
from random_data import sample_noisy_linear
from shared import cmap_wg, cmap_wr, rgb


jax.config.update("jax_enable_x64", True)


prior = Gaussian(mu=jnp.zeros(2), Sigma=jnp.eye(2))

phi = lambda x: jnp.hstack([jnp.ones_like(x), x])

key = jax.random.PRNGKey(0)
Xn, Yn = sample_noisy_linear(key, n=30, a=1.5, b=-0.5, sigma=0.3)

sigma = 1.5
# import ipdb; ipdb.set_trace()
posterior = prior.condition(phi(Xn[:, None]), Yn, sigma**2 * jnp.eye(len(Xn)))

x = jnp.linspace(-5, 5, 100)[:, None]
f_prior = phi(x) @ prior
f_posterior = phi(x) @ posterior


def regression_plot(prior: Gaussian, phi: callable, X, Y, sigma):
    fig, axis = plt.subplots(1, 2, sharex=True, sharey=True)
    for ax, g, cmap, color in zip(
        axis, [f_prior, f_posterior], [cmap_wg, cmap_wr], [rgb.tue_dark, rgb.tue_red]
    ):
        plot_gaussian(
            ax, g, x, cmap=cmap, color=color, yy=jnp.linspace(-10, 20, 300)[:, None]
        )
    for ax in axis:
        ax.plot(x, phi(x), color=rgb.tue_blue, label="feature", lw=0.75)
        ax.errorbar(
            X,
            Y,
            yerr=sigma * jnp.ones_like(Y),
            fmt="o",
            ms=2,
            color=rgb.tue_dark,
            label="dark",
        )
    ax.set_xlabel("$x$")
    ax.set_ylabel("$f(x), y")
    ax.set_ylim(-10, 20)
    ax.set_xlim(-5, 5)
    axis[0].set_title("prior")
    axis[1].set_title("posterior")
    return fig, axis


# Nonlinear regression
sigma_n = 0.4
N = Yn.shape[0]  # number of datapoints
regression_plot(prior, phi, Xn, Yn, sigma_n)
plt.show()

print("bla")
