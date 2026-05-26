from jax import numpy as jnp
from jax import random
import jax.random as jr


key = random.PRNGKey(0)


# Transition model
def f(xold: jnp.ndarray, key: random.key) -> jnp.ndarray:
    key,subkey=random.split(key)
    return xold +jr.normal(subkey,xold.shape)

# Observation model
def h(x: jnp.array, key: random.key) -> jnp.array:
    key,subkey = random.split(key)
    return x[0] + 0.1**2 * jr.normal(subkey, x[0].shape)


X = []
X.append(jnp.zeros(10))


for i in range(10):
    X.append(f(X[-1], key))