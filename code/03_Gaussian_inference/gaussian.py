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

    def prec_mult(self, x:Float[Array, "D "]) -> Float[Array, "D "]:
        """precision matrix multiplication implements Sigma^{-1}@x. For numerical stability, we us the Cholesky factorization"""

    def log_pdf(self, x: Float[Array, "D "]) -> float:
        return (
            -0.5 * (x - self.mu) @ jnp.linalg.solve(self.Sigma, x - self.mu)
            - 0.5 * jnp.linalg.slogdet(self.Sigma)[1]
            - 0.5 * len(self.mu) * jnp.log(2 * jnp.pi)
        )

    def pdf(self, x: Float[Array, "D "]) -> float:
        return jnp.exp(self.log_pdf(x))

    def precision(self):
        # Probably not want to use this directly, not stable and slow
        return jnp.linalg.inv(self.Sigma)

    def mp(self):
        return self.precision @ self.mu

    def __mul__(self, other: Self) -> Self:
        # new Cov = (A^-1 + B^-1)^-1
        Sigma = jnp.linalg.inv(self.precision + other.precision)
        mu = Sigma @ (self.mp + other.mp)
        return Gaussian(mu=mu, Sigma=Sigma)

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
