import os

os.environ["QT_QPA_PLATFORM"] = "xcb"
import matplotlib

matplotlib.use("TkAgg")  # Tk
import matplotlib.patches as patches
import matplotlib.pyplot as plt

# import pandas as pd
import jax
from jax import jit
from jax import numpy as jnp
from jax import random
from jax.scipy.linalg import cho_factor, cho_solve
from sklearn.datasets import make_moons
from tqdm import tqdm, trange
from matplotlib.colors import LinearSegmentedColormap
from probML2025.shared import rgb, cmap_wr, cmap_wd, cmap_bwo, KEY
from probML2025.gaussians import Gaussian

key = random.PRNGKey(0)


cmap = LinearSegmentedColormap.from_list(
    "owb", [rgb.tue_orange, (1, 1, 1), rgb.tue_blue], N=1024
).reversed()
dw = LinearSegmentedColormap.from_list("dw", [rgb.tue_dark, "w"], N=1024)
key = random.PRNGKey(0)

# here's the dataset:
X, y = make_moons(n_samples=100, noise=0.1, random_state=0)
N = X.shape[0]
ysigned = y.copy()
ysigned[y == 0] = -1

plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap, s=10, alpha=0.9, edgecolor="None")
plt.xlabel("$x_1$")
plt.ylabel("$x_2$")
# plt.show()

# we construct a ``random exponential family'' model with a Gaussian prior on the natural parameters.

# first, find the boundaries of the data:
xmin, xmax = X[:, 0].min(), X[:, 0].max()
ymin, ymax = X[:, 1].min(), X[:, 1].max()
center = 0.5 * jnp.array([xmin + xmax, ymin + ymax])
scale = jnp.array([xmax - xmin, ymax - ymin])

# put Gaussian features at random locations throughout the data domain:
phi_key, key = random.split(key)
n_features = 400

centers = (random.uniform(phi_key, shape=(n_features, 2)) - 0.5) * 2 * scale + center


def phi(x):
    """Gaussian features."""
    ell = 0.2
    return jnp.exp(
        -0.5 * jnp.sum((x[:, None, :] - centers[None, :, :]) ** 2 / ell**2, axis=-1)
    )


# link function:
def sigmoid(x):
    return 1.0 / (1.0 + jnp.exp(-x))


# Datengrid zur visualisierung
x1 = jnp.linspace(-3, 4, 50)
x2 = jnp.linspace(-2, 3, 50)
X1, X2 = jnp.meshgrid(x1, x2)
X_grid = jnp.stack([X1.flatten(), X2.flatten()], axis=1)

# N(theta; 0, I)
prior = Gaussian(mu=jnp.zeros(n_features), Sigma=jnp.eye(n_features))

# Sample mit 400 Dimensionen
sample = prior.sample(key, 1)

# phi~ = phi(X).T * theta~
phi_sample = phi(X_grid) @ sample.T

fig, ax = plt.subplots()

# ax.contourf(
#     X1, X2, (phi(X_grid) @ jnp.ones(n_features)).reshape(X1.shape),
#     cmap=cmap, alpha=0.5, vmax=1, vmin=0
# )

# Plot markers for the gaussian feature centers
ax.plot(
    centers[:, 0],
    centers[:, 1],
    "o",
    color=rgb.tue_gray,
    markersize=2,
    alpha=0.2,
    label="features",
)

# Plot the probability distribution, (contour-surface of the visualization-grid-test-points)
contour = ax.contourf(
    X1, X2, sigmoid(phi_sample).reshape(X1.shape), cmap=cmap, alpha=0.5, vmax=1, vmin=0
)

scatter = ax.scatter(
    X[:, 0], X[:, 1], c=y, cmap=cmap, s=10, alpha=1.0, edgecolor="None"
)

cb = fig.colorbar(ax=ax, mappable=scatter)
cb.set_ticks([0, 1])
plt.xlabel("$x_1$")
plt.ylabel("$x_2$")
# plt.show()


### Optimization setup


def log_likelihood(w, X, ysigned):
    # f(w,x) = phi(x).T * w
    # p(y|f) = 1 / (1 + exp(-yf))
    # log p(y|f) = 0 - log(exp(0) + exp(-yf))
    f = phi(X) @ w
    log_lik = jnp.sum(-jnp.logaddexp(0.0, -ysigned * f))
    pi = sigmoid(ysigned * f)
    g = jnp.sum(((1.0 - pi) * ysigned)[:, None] * phi(X), axis=0)
    # hessian = -(pi* (1.0-pi)) * phi(X).T @ phi(X)
    C = phi(X).T * jnp.sqrt(pi * (1.0 - pi))  # shape (n_features,N)
    return log_lik, g, C  # hessian = -C@C.T


from jax import grad, hessian


# INIT for Newton optimization
w = jnp.zeros(shape=(n_features,))
losses = []


V = prior.Sigma
Vi = prior.precision


@jit
def update(w, X, ysigned):
    log_lik, g, C = log_likelihood(w, X, ysigned)
    # The original Hessian can be computed via (C*C.T + V^-1) but it suffers often from very small eigenvalues
    # Intead of cov = C.T*C, I + C.T @ V @ C improves numerical stability
    # (C*C.T + V^-1)^-1 = V - V*C.T*B^-1*C*V can be achieved by applying the matrix inversion lemma (woodburry formula for updating inverses)
    B = jnp.eye(N) + C.T @ V @ C  # shape (n_features, n_features)
    scaling = V - V @ C @ jnp.linalg.solve(B, C.T @ V)  # solve only once,
    w = scaling @ (g + C @ C.T @ w)
    return w, log_lik, C


log_lik = -jnp.inf

# Newton Optimization, blazing fast, chakaa
while True:
    w, log_lik_new, C = update(w, X, ysigned)
    Delta = log_lik_new - log_lik
    log_lik = log_lik_new
    log_prior_new = prior.log_pdf(w)
    losses.append(-(log_lik + log_prior_new))
    if Delta < 1e-6:
        print(
            f"Converged with log likelihood {log_lik_new:.3f} after {len(losses)} iterations."
        )
        break


# Objects for Laplace Approximation
w_trained = w
hessian = C @ C.T + prior.precision()
posterior = Gaussian(mu=w_trained, Sigma=jnp.linalg.inv(hessian))  # linalg.inv slow :)

fig, ax = plt.subplots()
ax.plot(losses)
ax.axhline(0, color=rgb.tue_dark, lw=0.75)
ax.set_xlabel("epoch")
ax.set_ylabel("loss")
# plt.show()

# MAP
posterior_mode = phi(X_grid) @ w_trained.T
f_trained = phi(X) @ w_trained.T

fig, ax = plt.subplots()

contour = ax.contourf(
    X1,
    X2,
    sigmoid(posterior_mode).reshape(X1.shape),
    cmap=cmap,
    alpha=1,
    vmax=1,
    vmin=0,
    levels=100,
)

ax.scatter(
    X[:, 0],
    X[:, 1],
    c=sigmoid(f_trained),
    cmap=cmap,
    s=5,
    alpha=1.0,
    edgecolor=rgb.tue_dark,
)
ax.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap, s=10, alpha=0.5, edgecolor="None")

cb = fig.colorbar(ax=ax, mappable=scatter)
cb.set_ticks([0, 1])
ax.set_xlabel("$x_1$")
ax.set_ylabel("$x_2$")
# plt.show()

# Push Gaussian through phi(.)  [sampled]
samples = phi(X_grid) @ posterior.sample(key, 100).T
mean_prediction = sigmoid(samples).mean(axis=1)

posterior_std_dev = jnp.sqrt(
    jnp.sum((phi(X_grid) @ jnp.linalg.cholesky(posterior.Sigma).T) ** 2, axis=1)
)

posterior_std_dev = jnp.sqrt((phi(X_grid) @ posterior).Sigma.diagonal())
posterior_var = (phi(X_grid) @ posterior).Sigma.diagonal()
posterior_predictive = sigmoid(
    posterior_mode / jnp.sqrt(1 + jnp.pi / 8 * posterior_var)
)  # MacKey approximation

fig, axs = plt.subplots(1,6,sharex=True, sharey=True)

for i in range(3):
    axs[i].contourf(
        X1,
        X2,
        sigmoid(samples[:, i]).reshape(X1.shape),
        cmap=cmap,
        alpha=1,
        vmax=1,
        vmin=0,
        levels=100,
    )
    axs[i].set_xlabel("$x_1$")
    axs[i].set_title(f"sample {i+1}")

axs[3].contour = ax.contourf(
    X1,
    X2,
    posterior_predictive.reshape(X1.shape),
    cmap=cmap,
    alpha=1,
    vmax=1,
    vmin=0,
    levels=100,
)
axs[4].contour = ax.contourf(
    X1,
    X2,
    mean_prediction.reshape(X1.shape),
    cmap=cmap,
    alpha=1,
    vmax=1,
    vmin=0,
    levels=100,
)
axs[5].contour = ax.contourf(
    X1,
    X2,
    (sigmoid(posterior_mode) - posterior_predictive).reshape(X1.shape),
    cmap=cmap,
    alpha=1,
    vmax=1,
    vmin=0,
    levels=100,
)


for i in range(3):

    cb = fig.colorbar(ax=ax, mappable=scatter)
    cb.set_ticks([0, 1])
    axs[3].set_xlabel("$x_1$")
    axs[0].set_ylabel("$x_2$")
    axs[4].set_title(r"$\approx p(y\mid x)$")
    axs[4].set_title(r"$\approx p(y\mid x)$")
    axs[5].set_title(r"$\approx p(y\mid x) - \hat{p}(y\mid x)$")

    for ax in axs:
        ax.scatter(
            X[:, 0], X[:, 1], c=y, cmap=cmap, s=5, alpha=1.0, edgecolor=rgb.tue_dark
        )
plt.show()
