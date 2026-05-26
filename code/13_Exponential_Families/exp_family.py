import abc
import functools
from __future__ import annotations
from jax import numpy as jnp
from numpy.typing import ArrayLike


class ExponentialFamiliy(abc.ABC):
    @abc.abstractmethod
    def sufficient_statistics(self, x: ArrayLike | jnp.ndarray, /) -> jnp.ndarray:
        """Signature `(D)->(P)'"""

    @abc.abstractmethod
    def log_base_measure(self, x: ArrayLike | jnp.ndarray, /) -> jnp.ndarray:
        """Signature `(D)->()'"""

    @abc.abstractmethod
    def log_partition(
        self, natural_parameters: ArrayLike | jnp.ndarray, /
    ) -> jnp.ndarray:
        """Signature `(D)->()'"""

    def logpdf(
        self, x: ArrayLike | jnp.ndarray, natural_parameters: ArrayLike | jnp.ndarray, /
    ) -> jnp.ndarray:
        # log p(x|w) = log h(x) + sufficient_statistics(x) @ w - log Z(w)
        x = jnp.asarray(x)
        linear_term = (
            self.sufficient_statistics(x)[..., None, :] @ natural_parameters[..., None]
        )[..., 0, 0]

        return (
            self.log_base_measure(x)
            + linear_term
            - self.log_partition(natural_parameters)
        )

    def conjugate_log_partition(
        self, alpha: ArrayLike | jnp.ndarray, nu: ArrayLike | jnp.ndarray, /
    ) -> jnp.ndarray:  # (P),()->()
        """
        The log partition function of the conjugate exponential family
        """
        raise NotImplementedError()

    def conjugate_prior(self) -> "ConjugateFamily":
        return ConjugateFamily(self)

    def posterior_parameters(
        self,
        prior_natural_parameters: ArrayLike | jnp.ndarray,
        data: ArrayLike | jnp.ndarray,
    ) -> jnp.ndarray:  # (P),(D)->(P)
        """Computes the natural parameters of the posterior distribution under the conjugate prior.
        We could also call this method "learn", "infer", ...
        This can be implemented already in the abc and inherited by all subclasses, even if the conjugate log partition function is not available.
          (In the latter case, only the unnormalized posterior is immediately available, see below)
        """

        prior_natural_parameters = jnp.asarray(prior_natural_parameters)
        sufficient_statistic = self.sufficient_statistics(data)
        n = sufficient_statistic[..., 0].size
        expected_sufficient_statistics = jnp.sum(
            sufficient_statistic,
            axis=tuple(range(sufficient_statistic.ndim)),
        )
        alpha_prior, nu_prior = (
            prior_natural_parameters[:-1],
            prior_natural_parameters[-1],
        )
        return jnp.append(alpha_prior + expected_sufficient_statistics, nu_prior + n)



class ConjugateFamily(ExponentialFamiliy):
    def __init__(self, liklihood: ExponentialFamiliy) -> None:
        self._liklihood = liklihood

    def sufficient_statistics(self, w: ArrayLike | jnp.ndarray, /) -> jnp.ndarray: #(D)->(P)
        return jnp.append(w, -self._liklihood.log_partition(w),)

    def log_base_measure(self, w: ArrayLike|jnp.ndarray,/)-> jnp.ndarray:
        w = jnp.asarray(w)
        return jnp.zeros_like(w[...,0])
    
    def log_partition(self, natural_parameters: ArrayLike|jnp.ndarray,/) -> jnp.ndarray:
        natural_parameters = jnp.asarray(natural_parameters)
        alpha, nu= natural_parameters[:-1], natural_parameters[-1]
        return self._liklihood.conjugate_log_partition(alpha,nu)

    def unnormalized_logpdf(self, w:ArrayLike | jnp.ndarray, natural_parameters: ArrayLike | jnp.ndarray, /) -> jnp.ndarray:
        """If the log partition is not available, we can still compute the unnormalized log pdf"""
        return self.sufficient_statistics(w) @ jnp.asarray(natural_parameters)