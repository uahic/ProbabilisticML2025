import os
import jax
import functools
import requests
import pandas as pd
from jax import grad, hessian, jit
from scipy import optimize
from gaussians import plot_gaussian
from gp import GaussianProcess
from jax import numpy as jnp
from io import StringIO
from shared import rgb, cmap_wr, cmap_wd, cmap_bwo, KEY


os.environ["QT_QPA_PLATFORM"] = "xcb"
# os.environ["QT_QPA_PLATFORM"] = "wayland"        # must be before importing matplotlib
# os.environ["QT_LOGGING_RULES"] = "*.warning=false"  # suppress runtime warnings
import matplotlib

matplotlib.use("TkAgg")  # Tk
# matplotlib.use("QtAgg")   # Qt
# matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Load Data
url = "https://gml.noaa.gov/webdata/ccgg/trends/co2/co2_mm_mlo.csv"
s = requests.get(url).text
df = pd.read_csv(StringIO(s), sep=",", header=40, na_values="-999.99").dropna()
X = jnp.asarray(df["decimal date"])[:, None]
Y = jnp.asarray(df["average"])
N = len(X)
sigma = 0.1
print(f"Using {N} observations ranging from {X[0]} to {X[-1]}")


x = jnp.linspace(1930, 2053, 2000)[:, None]
yy = jnp.linspace(300, 490, 1024)[:, None]


def plot_data():
    """
    Plots the original dataset and setups the axes
    """
    fig, ax = plt.subplots()
    ax.plot(X, Y, label="data", color=rgb.tue_blue)

    ax.set_xlabel("Year")
    ax.set_ylabel("atmospheric CO$_2$ [ppm]")
    ax.xaxis.set_major_locator(plt.MultipleLocator(10))
    ax.xaxis.set_minor_locator(plt.MultipleLocator(1))
    ax.yaxis.set_major_locator(plt.MultipleLocator(20))
    ax.yaxis.set_minor_locator(plt.MultipleLocator(5))
    ax.set_xlim([1930, 2053])
    ax.set_ylim([300, 490])
    ax.grid(which="major", axis="both")
    return fig, ax


# Uncomment to show :)
# fig,ax = plot_data()
# plt.show()


# Naive approach (around ~07:00)
def linear_kernel(a, b, theta=1.0):
    """
    Linear function written in feature-transformed form:  phi(x).T * w = w1*x + w0
    Covariance as kernel using linear features:  phi(x1).T * Sigma * phi(x2) =  x1.T*x2 + 1
    """
    return theta**2 * (jnp.sum(a * b, axis=-1) + 1.0)


def zero_mean(x):
    return jnp.zeros_like(x[:, 0])


prior = GaussianProcess(zero_mean, lambda a, b: linear_kernel(a, b, theta=100.0))
posterior = prior.condition(Y, X, sigma)

# fig, ax = plot_data()
# plot_gaussian(ax, posterior(x), x, color=rgb.tue_red, yy=yy, cmap=cmap_wr, key=KEY)
# plt.show()


# More fancy with RQ/exponential kernel [~15:30]
def RQ_kernel(a, b, ell=1.0, alpha=1.0, theta=1.0):
    return theta**2 * (1 + jnp.sum((a - b) / ell, axis=-1) ** 2 / (2 * alpha) ** -alpha)


mean_Y = jnp.mean(Y)


def constant_mean(x):
    return mean_Y * jnp.ones_like(x[:, 0])


prior = GaussianProcess(
    constant_mean, functools.partial(RQ_kernel, ell=1.0, alpha=0.2, theta=3.0)
)
posterior = prior.condition(Y, X, sigma)

# TODO FIX ME, looks totally wrong, check if bug in GP or in plotting function occurs
# fig, ax = plot_data()
# plot_gaussian(ax, prior(x), x, color=rgb.tue_dark, yy=yy, cmap=cmap_wd, key=KEY)
# plot_gaussian(ax, posterior(x), x, color=rgb.tue_red, yy=yy, cmap=cmap_wr, key=KEY)
# plt.show()


def long_term_trend_kernel(a, b, theta=1.0, ell=20.0):
    return theta**2 * jnp.exp(-jnp.sum((a - b) ** 2, axis=-1) / (2 * ell**2))


prior = GaussianProcess(
    constant_mean, functools.partial(long_term_trend_kernel, ell=1.0, theta=20.0)
)

posterior = prior.condition(Y, X, sigma)

# TODO FIX ME, looks totally wrong, check if bug in GP or in plotting function occurs
# fig, ax = plot_data()
# plot_gaussian(ax, prior(x), x, color=rgb.tue_dark, yy=yy, cmap=cmap_wd, key=KEY)
# plot_gaussian(ax, posterior(x), x, color=rgb.tue_red, yy=yy, cmap=cmap_wr, key=KEY)
# plt.show()


@jit
def sum_kernel(
    a, b, theta_long=1.0, theta_rq=1.0, ell_long_term=20.0, ell=1.0, alpha=1.0
):
    long_term = long_term_trend_kernel(a, b, theta=theta_long, ell=ell_long_term)
    short_term = RQ_kernel(a, b, theta=theta_rq, ell=ell, alpha=alpha)
    return long_term + short_term


prior = GaussianProcess(
    constant_mean,
    functools.partial(
        sum_kernel,
        theta_long=100.0,
        theta_rq=5.0,
        ell_long_term=50.0,
        ell=1.0,
        alpha=0.2,
    ),
)
posterior = prior.condition(Y, X, sigma)

# TODO FIX ME, looks totally wrong, check if bug in GP or in plotting function occurs
# fig, ax = plot_data()
# plot_gaussian(ax, prior(x), x, color=rgb.tue_dark, yy=yy, cmap=cmap_wd, key=KEY)
# plot_gaussian(ax, posterior(x), x, color=rgb.tue_red, yy=yy, cmap=cmap_wr, key=KEY)
# plt.show()


def periodic_kernel(x, y, period=1.0, ell=1.0, theta=1.0):
    return theta**2 * jnp.exp(
        -2 * jnp.sin(jnp.pi * jnp.sum(x - y, axis=-1) / period) ** 2 / ell
    )


prior = GaussianProcess(constant_mean, lambda a, b: periodic_kernel(a, b))

decayingprior = GaussianProcess(
    constant_mean,
    lambda a, b: periodic_kernel(a, b) * long_term_trend_kernel(a, b, ell=10, theta=1),
)

fix, axs = plt.subplots(1, 2)
x_periodic = jnp.linspace(0, 10, 400)[:, None]
yy_periodic = jnp.linspace(357, 363, 1024)[:, None]

# for p, ax in zip([prior, decayingprior], axs):
#     plot_gaussian(
#         ax,
#         p(x_periodic),
#         x_periodic,
#         color=rgb.tue_dark,
#         num_samples=2,
#         cmap=cmap_wd,
#         key=KEY,
#     )
# plt.show()


@jit
def sum_kernel(
    x, y, theta_long=1.0, theta_periodic=1.0, ell_long_term=20.0, ell_periodic=1.0
):
    long_term = long_term_trend_kernel(x, y, theta=theta_long, ell=ell_long_term)
    periodic = periodic_kernel(x, y, theta=theta_periodic, ell=ell_periodic, period=1.0)
    return long_term + periodic


prior = GaussianProcess(
    constant_mean,
    functools.partial(
        sum_kernel,
        theta_long=100.0,
        theta_periodic=5.0,
        ell_long_term=50.0,
        ell_periodic=1.0,
    ),
)
posterior = prior.condition(Y, X, sigma)

# TODO FIX ME, looks totally wrong, check if bug in GP or in plotting function occurs
# fig, ax = plot_data()
# plot_gaussian(ax, prior(x), x, color=rgb.tue_dark, yy=yy, cmap=cmap_wd, key=KEY)
# plot_gaussian(ax, posterior(x), x, color=rgb.tue_red, yy=yy, cmap=cmap_wr, key=KEY)
# plt.show()


# Ok, time to setup a model [~41:31], he recreates some kernels here with different parameters
def long_term_trend_kernel(x, y, theta=100.0, ell=100.0):
    return theta**2 * jnp.exp(-jnp.sum((x - y) ** 2, axis=-1) / (2 * ell**2))


def periodic_kernel(x, y, ell_period=1.0, ell_decay=50.0, theta=1.0):
    period = 1.0  # year
    return theta**2 * jnp.exp(
        -2
        * jnp.sin((jnp.pi * jnp.sum(x - y, axis=-1) / period) ** 2 / ell_period**2)
        * jnp.exp(-jnp.sum((x - y) ** 2, axis=-1) / (2 * ell_decay**2))
    )


def mid_term_trend(x, y, ell=1.0, alpha=1.0, theta=1.0):
    return theta**2 * (
        1 + jnp.sum((x - y) ** 2, axis=-1) / (2 * alpha * ell**2) ** (-alpha)
    )


def noise_kernel(x, y, ell=1.0, theta_weather=0.1, theta_measurement=0.1):
    return theta_weather**2 * jnp.exp(
        -jnp.sum((x - y) ** 2, axis=-1) / (2 * ell**2)
        + theta_measurement**2 * jnp.all(x == y, axis=-1)
    )


@jit
def model_kernel(x, y, parameters):
    (
        theta_long,
        ell_long_term,
        theta_periodic,
        ell_decay_periodic,
        ell_periodic,
        theta_mid_term,
        ell_mid_term,
        shape_mid_term,
        theta_weather,
        ell_weather,
        theta_measurement,
    ) = parameters
    return (
        long_term_trend_kernel(x, y, theta=theta_long, ell=ell_long_term)
        + periodic_kernel(
            x,
            y,
            theta=theta_periodic,
            ell_period=ell_periodic,
            ell_decay=ell_decay_periodic,
        )
        + mid_term_trend(
            x, y, theta=theta_mid_term, ell=ell_mid_term, alpha=shape_mid_term
        )
        + noise_kernel(
            x,
            y,
            ell_weather,
            theta_weather=theta_weather,
            theta_measurement=theta_measurement,
        )
    )


theta_long = 100.0  # ppm
ell_long_term = 100.0  # years
theta_periodic = 2.0  # ppm
ell_decay_periodic = 2.0  # ppm
ell_periodic = 1.0  # years
theta_mid_term = 1.0  # ppm
ell_mid_term = 1.0  # years
shape_mid_term = 1.0  # unitless
theta_weather = 0.1  # ppm
ell_weather = 0.1  # years
theta_measurement = 0.1  # ppm

init_params = jnp.asarray(
    [
        theta_long,
        ell_long_term,
        theta_periodic,
        ell_decay_periodic,
        ell_periodic,
        theta_mid_term,
        ell_mid_term,
        shape_mid_term,
        theta_weather,
        ell_weather,
        theta_measurement,
    ]
)

# define the model
gp = GaussianProcess(
    constant_mean, functools.partial(model_kernel, parameters=init_params)
)

# condition the prior on the data
gp_posterior = gp.condition(Y, X, sigma)

# fig, ax = plot_data()
# plot_gaussian(ax, gp(x), x, color=rgb.tue_dark, yy=yy, cmap=cmap_wd, key=KEY)
# plot_gaussian(ax, gp_posterior(x), x, color=rgb.tue_red, yy=yy, cmap=cmap_wr, key=KEY)
# plt.show()

# samples = gp(X).sample(key=KEY, num_samples=7)
# fig, ax = plot_data()
# ax.plot(X, samples.T, color=rgb.tue_blue, alpha=1.0)
# ax.set_ylim([jnp.min(samples[:]), jnp.max(samples[:])])


# Hyperparameter optimization [Video 08 - around 1:11:00]
def NegEvidence(params):
    gp = GaussianProcess(
        constant_mean,
        functools.partial(model_kernel, parameters=params),
    )
    # gp(X) just instantiates Gaussian(mu=constant_mean(X), sigma=model_kernel(X, **params)) here (just a linear transform of a Gaussian).
    # log_pdf(Y) to get the Gaussian in log-form to pass it to an optimizer (*-1 to pose it as minimization problem)
    return -gp(X).log_pdf(Y)


params = init_params

print("Running CG optimizer, this can take some minutes :-)")
results = optimize.minimize(
    NegEvidence,
    params,
    method="Newton-CG",
    jac=jax.grad(NegEvidence),
    hess=jax.hessian(NegEvidence),
    options={"xtol": 1e-6, "disp": True, "maxiter": 300},
)


# Parameters of Rassmussen & Williams to compare our optimized params to
RW_params = [66.0, 67.0, 2.4, 90.0, 1.3, 0.66, 1.2, 0.78, 0.18, 0.13, 0.19]

units = [
    "ppm",
    "years",
    "ppm",
    "years",
    "years",
    "ppm",
    "years",
    "",
    "ppm",
    "years",
    "ppm",
]
names = [
    "theta_long",
    "ell_long_term",
    "theta_periodic",
    "ell_decay_periodic",
    "ell periodic",
    "theta_mid_term",
    "ell_mid_term",
    "shape_mid_term",
    "theta_weather",
    "ell_weather",
    "theta_measurement",
]

for p, name, r, i, u in zip(results.x, names, RW_params, init_params, units):
    print(
        name.rjust(25)
        + f"{i:9.2f}"
        + f" -> "
        + f"{p:6.2f}"
        + 5 * " "
        + "Ref: "
        + f"{r:6.2f}"
        + " "
        + u
    )

print(f"Initial Evidence: {NegEvidence(init_params):.2f}")
print(f"Optimized Evidence: {NegEvidence(results.x):.2f}")
print(f"Reference Evidence: {NegEvidence(RW_params):.2f}")


opt_params = jnp.asarray(results.x)

gp = GaussianProcess(
    constant_mean, functools.partial(model_kernel), parameters=opt_params
)

gp_posterior = gp.condition(Y, X, sigma)

samples = gp(X).sample(key=KEY, num_samples=7)
fig, ax = plot_data()
ax.plot(X, samples.T, color=rgb.tue_blue, alpha=1.0)
ax.set_ylim([jnp.min(samples[:]), jnp.max(samples[:])])
ax.set_title("can you pick out the data?")


fig, ax = plot_data()
plot_gaussian(
    ax, gp(x), x, color=rgb.tue_dark, yy=yy, cmap=cmap_wd, key=KEY, num_samples=10
)
plot_gaussian(ax, gp_posterior(x), x, color=rgb.tue_red, yy=yy, cmap=cmap_wr, key=KEY)


hess_at_reference = jnp.array(jax.hessian(RW_params))
hess_at_opt = jnp.array(jax.hessian(results.x))

fig, ax = plt.subplots(1, 2)
f_ref = ax[0].imshow(hess_at_reference, cmap=cmap_bwo, vmin=-0.1, vmax=0.1)
ax[0].set_title("Hessian at reference")
f_opt = ax[1].imshow(hess_at_opt, cmap=cmap_bwo, vmin=-0.1, vmax=0.1)
ax[1].set_title("Hessian at optimized")
fig.colorbar(f_ref, ax=ax[1])
plt.show()


approximate_covariance = jnp.linalg.inv(hess_at_opt)
error_bars = 2 * jnp.sqrt(jnp.diag(approximate_covariance))

print("=" * 80)
for p, name, r, e, i, u in zip(
    results.x, names, RW_params, error_bars, init_params, units
):
    print(
        name.rjust(25)
        + f"{i:9.2f}"
        + f" -> "
        + f"{p:6.2f}"
        + " +/- "
        + f"{p:6.2f}"
        + 5 * " "
        + "Ref: "
        + f"{r:6.2f}"
        + " "
        + u
    )
print("=" * 80)
