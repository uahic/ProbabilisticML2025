from jax import numpy as jnp
from jaxtyping import Array, Float

import dataclasses
import functools
import jax

from typing import Self
from jax import numpy as jnp
from jaxtyping import Array, Float, Int, PRNGKeyArray, PyTree


jax.config.update("jax_enable_x64", True)

# Gauss(x; a,A)*Gauss(x; b, B) = Gauss(x; c, C) * Z
# C = (A^{-1} + B^{-1})^{-1}
# c = C* (A^{-1}*a + B^{-1}*b)
# Z = Gauss(a; b, A+B)


@dataclasses.dataclass
class Gaussian:
    mu: Float[Array, "D "]
    Sigma: Float[Array, "D D"]

    @functools.cached_property
    def cov_SVD(self):
        """square root of the cov matrix via SVD"""
        if jnp.isscalar(self.mu):
            return jnp.eye(1), jnp.sqrt(self.Sigma).reshape(1, 1)
        else:
            Q, S, _ = jnp.linalg.svd(self.Sigma, full_matrices=True, hermitian=True)
            return Q, jnp.sqrt(S)

    @functools.cached_property
    def logdet(self):
        """log-det of the cov matrix e.g. for computing the log-pdf"""
        _, S = self.cov_SVD
        return 2 * jnp.sum(jnp.log(S))

    @functools.cached_property
    def precision(self):
        """precision matrix you probably dont want to use this directly, rather prec_mult"""
        Q, S = self.cov_SVD
        return Q @ jnp.diag(1 / S) ** 2 @ Q.T

    @functools.cached_property
    def std(self) -> Float[Array, "D "]:
        """standard deviation"""
        if jnp.isscalar(self.mu):
            return jnp.sqrt(self.Sigma)
        else:
            return jnp.sqrt(self.Sigma.diagonal())

    def prec_mult(self, x: Float[Array, "D "]) -> Float[Array, "D "]:
        """precision matrix multiplication implements Sigma^{-1}@x. For numerical stability, we us the Cholesky factorization"""
        Q, S = self.cov_SVD
        return Q @ jnp.diag(1 / S**2) @ Q.T @ x

    @functools.cached_property
    def mp(self):
        """precision-adjusted mean"""
        return self.precision @ self.mu

    def log_pdf(self, x: Float[Array, "D "]) -> float:
        """log N(x;mu,Sigma)"""
        if len(self.mu) == 1:
            return (
                -0.5 * (x - self.mu) ** 2 / self.Sigma
                - 0.5 * jnp.log(self.Sigma)
                - 0.5 * jnp.log(2 * jnp.pi)
            )
        else:
            return (
                -0.5 * (x - self.mu).T @ self.prec_mult(x - self.mu)
                - 0.5 * self.logdet
                - 0.5 * len(self.mu) * jnp.log(2 * jnp.pi)
            )

    def pdf(self, x: Float[Array, "D "]) -> float:
        """N(x;mu,Sigma)"""
        return jnp.exp(self.log_pdf(x))

    def cdf(self, x):
        if jnp.isscalar(self.mu):
            return 0.5 * (
                1 + jax.scipy.special.erf((x - self.mu) / jnp.sqrt(2 * self.Sigma))
            )
        else:
            raise NotImplementedError(
                "CDF for multivariate Gaussian is not implemented"
            )

    def precision(self):
        # Probably not want to use this directly, not stable and slow
        return jnp.linalg.inv(self.Sigma)

    def __mul__(self, other: Self, return_log_normalizer: bool = False) -> Self:
        """Products of Gaussian pdfs are Gaussian pdfs!
        Multiplication of two Gaussian PDFs (not RVs)
        other: Gaussian RV
        """
        Sigma = jnp.linalg.inv(self.precision + other.precision)
        mu = Sigma @ (self.mp + other.mp)
        if return_log_normalizer:
            Z = Gaussian(mu=other.mu, Sigma=self.Sigma + other.Sigma).log_pdf(self.mu)
            return Gaussian(mu=mu, Sigma=Sigma), Z
        else:
            return Gaussian(mu=mu, Sigma=Sigma)

    def __rmatmul__(self, A: Float[Array, "N D"]) -> Self:
        """Linear Maps of Gaussian RVs are Gaussian RVs
        N(x; A*mu, A*Sigma*A.T)
        """
        return Gaussian(mu=A @ self.mu, Sigma=A @ self.Sigma @ A.T)

    def __getitem__(self, i) -> Self:
        """Marginals"""
        return Gaussian(
            mu=jnp.atleast_1d(self.mu[i]), Sigma=jnp.atleast_2d(self.Sigma[i, i])
        )

    # Shift Gaussian
    @functools.singledispatchmethod
    def __add(self, other: Float[Array, "D "] | float) -> Self:
        """
        Affine maps of Gaussian RVs are Gaussian RVs
        shift of a Gaussian RV by a constant
        We implement this as a singledispatchmethod, because jnp.ndarrays can not be dispatched on and register the addition of two RVs below
        """
        other = jnp.asarray(other)
        return Gaussian(mu=self.mu + other, Sigma=self.Sigma)

    def condition(
        self, A: Float[Array, "N D"], y: Float[Array, "N"], Lambda: Float[Array, "N N"]
    ) -> Self:
        """Linear conditionals of Gaussian RVs are Gaussian RVs
        return p(self | y) = N(y; A @ self, Lambda) * self / p(y)
        """
        Gram = A @ self.Sigma @ A.T + Lambda
        L = jax.scipy.linalg.cho_factor(Gram, lower=True)
        mu = self.mu + self.Sigma @ A.T @ jax.scipy.linalg.cho_solve(L, y - A @ self.mu)
        Sigma = self.Sigma - self.Sigma @ A.T @ jax.scipy.linalg.cho_solve(
            L, A @ self.Sigma
        )
        return Gaussian(mu=mu, Sigma=Sigma)

    def sample(self, key: PRNGKeyArray, num_samples=1):
        d = self.mu.shape[0]
        L = jnp.linalg.cholesky(self.Sigma)
        z = jax.random.normal(key, shape=(num_samples, d))
        return self.mu + z @ L.T


# @Gaussian.__add__.register
# def _add_gaussians(self, other: Gaussian) -> Gaussian:
#     # sum of the two Gaussian RVs
#     return Gaussian(mu=self.mu + other.mu, Sigma=self.Sigma + other.Sigma)


### Plotting
def gp_shading(yy, g: Gaussian) -> Float[Array, "N M"]:
    return jnp.exp(-((yy - g.mu) ** 2) / (2 * g.std**2))


def plot_gaussian(
    ax,
    g: Gaussian,
    xplot: Float[Array, "N"],
    color="C0",
    yy=None,
    cmap="viridis",
    key=jax.random.PRNGKey(0),
    **kwargs
) -> None:
    ax.plot(xplot, g.mu, color=color, lw=1, **kwargs)
    ax.plot(xplot, g.mu + 2 * g.std, color=color, **kwargs, lw=0.5, ls="--")
    ax.plot(xplot, g.mu - 2 * g.std, color=color, **kwargs, lw=0.5, ls="--")
    ax.plot(xplot, g.sample(key, num_samples=3).T, color=color, **kwargs, lw=0.5)

    if yy is not None:
        shading = gp_shading(yy, g)
        ax.imshow(
            shading,
            extent=(xplot[0, 0], xplot[-1, 0], jnp.min(yy), jnp.max(yy)),
            aspect="auto",
            origin="lower",
            cmap=cmap,
            alpha=0.8,  # shading/jnp.max(shading)
        )
        ax.set_ylim(jnp.min(yy), jnp.max(yy))
