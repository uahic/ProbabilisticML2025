from jax import numpy as jnp
from jaxtyping import Array, Float

def Bayes(joint: Float[Array, "d_x d_y"]) -> Float[Array, "d_x d_y"]:
    prior_X = jnp.sum(joint, axis=1) # P[X=x]
    evidence_Y = jnp.sum(joint, axis=0) # P[Y=y]

    likelihood_for_X_of_Y = joint/prior_X[:, None] # P[Y=y|X=x] = p[X=x, Y=y]/p[X=x]
    posterior_X_given_Y = (likelihood_for_X_of_Y * prior_X[:, None]) / evidence_Y[None, :]

    return prior_X, evidence_Y, likelihood_for_X_of_Y, posterior_X_given_Y

