from jax import numpy as jnp
from jaxtyping import Array, Float

import dataclasses
import functools
import jax

from typing import Self, Callable
from jax import numpy as jnp

from gaussians import Gaussian


jax.config.update("jax_enable_x64", True)

# Gauss(x; a,A)*Gauss(x; b, B) = Gauss(x; c, C) * Z
# C = (A^{-1} + B^{-1})^{-1}
# c = C* (A^{-1}*a + B^{-1}*b)
# Z = Gauss(a; b, A+B)


# This code is also equivalent the lazy generalization to the mathematical definition of a Gaussian Process.


# p(y | f(X)) = N(y; f_{x}, sigma^{2}*I) = Product^{n}_{i=1} N(y_{i]}; f(x_{i}), sigma^2)
@dataclasses.dataclass
class GaussianProcess:
    # mean function
    m: Callable[[jnp.ndarray], jnp.ndarray]  # maps |X^n -> |R^n

    # covariance function (mercer kernel)
    k: Callable[
        [jnp.ndarray, jnp.ndarray], jnp.ndarray
    ]  # maps ( |X x |X)^n -> |R^{n x n}

    def __call__(self, x):
        # the notation x[:, None, :] and x[None, :, :] is basically numpy/jax broadcasting of values into column and row vector.
        # the third dimension is necessary if each value of x has multiple features
        return Gaussian(mu=self.m(x), Sigma=self.k(x[:, None, :], x[None, :, :]))

    def condition(self, y, X, sigma):
        return ConditionalGaussianProcess(
            self, y, X, Gaussian(mu=jnp.zeros_like(y), Sigma=sigma * jnp.eye(len(y)))
        )


class ParametricGaussianProcess(GaussianProcess):

    def __init__(self, phi: Callable[[jnp.ndarray], jnp.ndarray], prior: Gaussian):
        self.phi = phi
        self.prior = prior
        super().__init__(self._mean, self._covariance)

    def _mean(self, x):
        x = jnp.asarray(x)
        return self.phi(x) @ self.prior.mu

    def _covariance(self, x1, x2):
        return self.phi(x1).squeeze() @ self.prior.Sigma @ self.phi(x2).squeeze().T


def gp_shading(yy, g: Gaussian) -> Float[Array, "N M"]:
    return jnp.exp(-((yy - g.mu) ** 2) / (2 * g.std**2))


def GP_Plot(
    gp: GaussianProcess,
    ax,
    yrange,
    yres=1000,
    color="C0",
    mean_kwargs={},
    std_kwargs={},
) -> None:
    pass
    # ax.plot(xplot, g.mu, color=color, lw=1, **kwargs)
    # ax.plot(xplot, g.mu + 2 * g.std, color=color, **kwargs, lw=0.5, ls="--")
    # ax.plot(xplot, g.mu - 2 * g.std, color=color, **kwargs, lw=0.5, ls="--")
    # ax.plot(xplot, g.sample(key, num_samples=3).T, color=color, **kwargs, lw=0.5)

    # if yy is not None:
    #     shading = gp_shading(yy, g)
    #     ax.imshow(
    #         shading,
    #         extent=(xplot[0, 0], xplot[-1, 0], jnp.min(yy), jnp.max(yy)),
    #         aspect="auto",
    #         origin="lower",
    #         cmap=cmap,
    #         alpha=0.8,  # shading/jnp.max(shading)
    #     )
    #     ax.set_ylim(jnp.min(yy), jnp.max(yy))


class ConditionalGaussianProcess(GaussianProcess):

    def __init__(self, prior: GaussianProcess, y, X, epsilon: Gaussian):
        self.prior = prior
        self.y = jnp.atleast_1d(y)  # (nsamples, )
        self.X = jnp.atleast_2d(X)  # (n_samples, n_samples)
        self.epsilon = epsilon
        super().__init__(self._mean, self._covariance)

    @functools.cached_property
    def predictive_covariance(self):
        return self.prior.k(self.X[:, None, :], self.X[None, :, :] + self.epsilon.Sigma)

    @functools.cached_property
    def predictive_covariance_cho(self):  # L
        return jax.scipy.linalg.cho_factor(self.predictive_covariance)

    @functools.cached_property
    def representer_weights(self):  # alpha
        return jax.scipy.linalg.cho_solve(
            self.predictive_covariance_cho,
            self.y - self.prior(self.X).mu - self.epsilon.mu,
        )
