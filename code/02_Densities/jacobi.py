from jax import numpy as jnp

n = 3  # Positive cases
m = 6  # negative cases

# x = wahrscheinlichkeit positiver case


x = jnp.linspace(0, 1, 10000)  # Diskretisierter Wkt. Raum.

posterior = x**n * (1 - x) ** m * 1  # p(n,m|x)p(x) with p(x)=1
posterior /= jnp.sum(posterior) / len(x)  # \int p(n,m|x)p(x)dx ~ \sum_i  p(n,m|xi)p(xi)
# where p(xi) = 1 / len(x) for uniform prior

"""
Wenn wir die Bucketgroessen von x aendern (also statt 10000 z.b. 500 oder so), dann erhalten wir auch einen anderen Eventraum.
"""


# Flexible diskretisierungen:

a0, b0, n, m, D = 0.1, 0.1, 1e3, 10, 100  # parameters of the experiment

# 2 Grids:
# 1) Lineare gleichmaessige Unterteilung
# 2) sym-log space aufteilung (viele Buckets am Rand gestaucht). Hier auch mit Reparametrisierung des Intervals [0,1] zu [-3,3]
grids = [
    jnp.linspace(0, 1, D),
    1 / (1 + 10 ** (-jnp.linspace(-3, 3, D))),
]  # grids[1]: sym-log scale
# import ipdb; ipdb.set_trace()

for grid in grids:
    widths = grid[1:] - grid[:-1]  # width of the bin
    locs = grid[:-1] + widths / 2
    posterior = locs ** (a0 + n - 1) * (1 - locs) ** (b0 + m - 1)
    posterior /= jnp.sum(posterior * widths)  # normalize
    print(
        f"Probability in the last (right-most) bucket: {posterior[-1]*widths[-1] = :.2f}"
    )


# How to fix?
# -> Change of Variable for Probability Density Functions

# Grundidee:
# Stell dir Koordinatensystem mit X,Y vor (c1 < x < c2) und eine monoton-steigend Funktion
# Y = u(X). Im Beispiel oben ist u(X) = symLog(X) mit einer inversen Funktion X = v(Y), dann ist die PDF von Y:

# p_{Y}(y) = p_{X}(v(y)) * | dv(y)/dy | = p_{X}(v(y)) * |du(x)/dx|^{-1}


from jax import grad, vmap
from jax.scipy.special import betaln
from jaxtyping import Float


def symlog(x: Float) -> Float:
    return 1 / (1 + 10 ** (-x))


grad_symlog = vmap(grad(symlog))


def p_x(x: Float, a: Float, b: Float) -> Float:
    log_p = (a - 1) * jnp.log(x) + (b - 1) * jnp.log(1 - x) - betaln(a, b)
    return jnp.exp(log_p)


def p_y(y: Float, a: Float, b: Float) -> Float:
    return p_x(symlog(y), a, b) * jnp.abs(grad_symlog(y))
