from jax import numpy as jnp
from jaxtyping import Array, Float

import dataclasses
import functools
import jax

from typing import Self, Callable
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
    def __add__(self, other: Float[Array, "D "] | float) -> Self:
        """
        Affine maps of Gaussian RVs are Gaussian RVs
        shift of a Gaussian RV by a constant
        We implement this as a singledispatchmethod, because jnp.ndarrays can not be dispatched on and register the addition of two RVs below
        """
        other = jnp.asarray(other)
        return Gaussian(mu=self.mu + other, Sigma=self.Sigma)

    def sample(self, key: PRNGKeyArray, num_samples: int = 1) -> Float[Array, "N D"]:
        if jnp.isscalar(self.mu):
            return jax.random.normal(key, (num_samples,)) * self.std + self.mu
        else:
            Q, S = self.cov_SVD
            z = S[..., :] * jax.random.normal(key, (num_samples, len(self.mu)))
            return z @ Q.T + self.mu

    def condition(
        self, A: Float[Array, "N D"], y: Float[Array, "N"], Lambda: Float[Array, "N N"]
    ) -> Self:
        """Linear conditionals of Gaussian RVs are Gaussian RVs
        return p(self | y) = N(y; A @ self, Lambda) * self / p(y)
        """

        import ipdb; ipdb.set_trace()
        Gram = A @ self.Sigma @ A.T + Lambda
        L = jax.scipy.linalg.cho_factor(Gram, lower=True)
        mu = self.mu + self.Sigma @ A.T @ jax.scipy.linalg.cho_solve(L, y - A @ self.mu)
        Sigma = self.Sigma - self.Sigma @ A.T @ jax.scipy.linalg.cho_solve(
            L, A @ self.Sigma
        )
        return Gaussian(mu=mu, Sigma=Sigma)

    def condition_pls(
        self,
        A: Float[Array, "N D"],
        y: Float[Array, "N"],
        Lambda: Float[Array, "N N"],
        max_steps=None,
        policy=None,
        atol=1e-6,
        rtol=1e-6,
    ) -> Self:
        """
        Conditioning the Gaussian using the probabilistic linear solver
        """
        N, M = A.shape
        assert y.shape == (N,)
        assert Lambda.shape == (N, N)
        assert self.mu.shape == (M,)

        Gram = A @ self.Sigma @ A.T + Lambda  # Shape (N,N)
        cov_pred_obs = self.Sigma @ A.T
        b = y - A @ self.mu

        # solving with prob linear solver
        solver_prior = Gaussian(mu=jnp.zeros((N,)), Sigma=jnp.eye(N))
        solve, _, _, _, _ = GEPNF(
            Gram,
            solver_prior,
            b=b,
            max_steps=max_steps,
            policy=policy,
            atol=atol,
            rtol=rtol,
        )

        posterior_mu = self.mu + cov_pred_obs @ solve(b).mu
        cov_correction = jnp.array(
            [solve(cov_pred_obs[i, :]).mu for i in range(cov_pred_obs.shape[0])]
        )
        posterior_Sigma = self.Sigma - cov_pred_obs @ cov_correction.T
        return Gaussian(mu=posterior_mu, Sigma=posterior_Sigma)

    # def sample(self, key: PRNGKeyArray, num_samples=1):
    #     d = self.mu.shape[0]
    #     L = jnp.linalg.cholesky(self.Sigma)
    #     z = jax.random.normal(key, shape=(num_samples, d))
    #     return self.mu + z @ L.T


# @Gaussian.__add__.register
# def _add_gaussians(self, other: Gaussian) -> Gaussian:
#     # sum of the two Gaussian RVs
#     return Gaussian(mu=self.mu + other.mu, Sigma=self.Sigma + other.Sigma)


class CGPolicy:
    def __call__(self, *, r, **kwargs):
        return -r


class CholeskyPolicy:
    def __init__(self, pivoted=True):
        self._pivoted = pivoted

    def __call__(self, *, i, r, **kwargs):
        s = jnp.zeros_like(r)
        return s.at[jnp.abs(r).argmax() if self._pivoted else i].set(1)


def GEPNF(
    A: Float[Array, "M N"],
    prior: Gaussian,
    b: Float[Array, "M"],
    max_steps=None,
    policy=None,
    atol=1e-6,
    rtol=1e-6,
) -> tuple[Callable, Float[Array, "M M"], Float[Array, "M N"], Float[Array, "M M"]]:
    """
    Probabilistic linear solver

    Solves the linear system A.T = b for x, where
    A: matrix of size (M, N)
    prior: Gaussian prior over the solution x of size (N,)
    b: right-hand size of size (M,). Used only for the solve poliy (can be set to a random vector if no rhs is given. Not used by all policies)
    max_steps: maximum number of steps taken in solve
    policy: policy choosing projections of b and A during the solve
        - e.g. CG Poliy, Cholesky Policy
    atol: absolute tolerance
    rtol: relative rolterance
    solver stops when the residual norm is below tol = atol + rtol * || b - A.T@mu ||_{2}

    Returns:
        solve(x:Float[Array, "N]) -> Gaussian: function that solves the linear system A.T*x = b for x
        R: matrix of size (M,M) containing the decomposition
        U: matrix of size (M,N) containing the decomposition
        S: matrix of size (N,M) containing the search directions
        # it holds the prior.Sigma^{-1} U @ R = A @ S
        Sigma: covariance matrix of posterior over x
    """

    M, N = A.shape
    if max_steps is None:
        max_steps = N
    else:
        max_steps = min(max_steps, N)

    if policy is None:
        policy = CholeskyPolicy()

    mu = prior.mu.copy()
    Sigma = prior.Sigma.copy()

    # residuals
    r = b - A.T @ mu
    r_norm = jnp.linalg.norm(r)
    tol = atol + rtol * r_norm

    # storage
    U, R, S, nu = (
        jnp.zeros((M, max_steps)),
        jnp.zeros((max_steps, max_steps)),
        jnp.zeros((N, max_steps)),
        jnp.zeros(max_steps),
    )

    alpha = jnp.zeros((max_steps,))

    n = 0

    while n < max_steps and r_norm > tol:
        s = policy(i=n, r=r)
        S = S.at[:, n].set(s)

        yn = A @ s
        un = Sigma @ yn
        un = un / jnp.sqrt(jnp.dot(yn, un))
        U = U.at[:, n].set(un)
        R = R.at[: n + 1, n].set(yn.T @ U[:, : n + 1])
        nu = nu.at[n].set(jnp.dot(yn, prior.mu))
        Sigma = Sigma - jnp.outer(un, un)

        # update the mean
        bn = jnp.dot(S[:, n], b)
        alpha = alpha.at[n].set((bn - nu[n] - jnp.dot(alpha[:n], R[:n, n])) / R[n, n])
        mu = mu + U[:, n] * alpha[n]
        r = b - A.T @ mu
        r_norm = jnp.linalg.norm(r)
        n += 1

    def solve(b: Float[Array, "N"]) -> Gaussian:
        alpha = jnp.zeros((max_steps,))
        if max_steps > 0:
            b0 = jnp.dot(S[:, 0], b)
            alpha = alpha.at[0].set((b0 - nu[0]) / R[0, 0])
        for n in range(1, max_steps):
            bn = jnp.dot(S[:, n], b)
            alpha = alpha.at[n].set(
                (bn - nu[n] - jnp.dot(alpha[:n], R[:n, n])) / R[n, n]
            )
        return Gaussian(prior.mu + U @ alpha, Sigma)

    return solve, R, U, S, Sigma


def GEQRF(
    A: Float[Array, "M N"], prior: Gaussian, max_steps=None
) -> tuple[Callable, Float[Array, "M M"], Float[Array, "M N"], Float[Array, "M M"]]:
    M, N = A.shape
    if max_steps is None:
        max_steps = N
    else:
        max_steps = min(max_steps, N)
    U, R, nu = (
        jnp.zeros((M, max_steps)),
        jnp.zeros((max_steps, max_steps)),
        jnp.zeros(max_steps),
    )
    Sigma = prior.Sigma

    for n in range(max_steps):
        an = A[:, n]
        un = Sigma @ an
        un = un / jnp.sqrt(jnp.dot(an, un))
        U = U.at[:, n].set(un)
        R = R.at[: n + 1, n].set(an.T @ U[:, : n + 1])
        nu = nu.at[n].set(jnp.dot(an, prior.mu))
        Sigma = Sigma - jnp.outer(un, un)

    def solve(b: Float[Array, "N"]) -> Gaussian:
        alpha = jnp.zeros((max_steps,))
        if max_steps > 0:
            alpha = alpha.at[0].set((b[0] - nu[0]) / R[0, 0])
        for n in range(1, max_steps):
            alpha = alpha.at[n].set(
                (b[n] - nu[n] - jnp.dot(alpha[:n], R[:n, n])) / R[n, n]
            )
        return Gaussian(prior.mu + U @ alpha, Sigma)

    return solve, U, R, Sigma


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
