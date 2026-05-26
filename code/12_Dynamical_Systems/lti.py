import numpy as np
from jax import numpy as jnp
from scipy.linalg import expm


def lti_disc(F, L, dt=1):
    """
    Discrete LTI ODE with Gaussian Noise
    Copyright Simo Saerkkae

    Syntax:
    [A,Q] = lti_disc(F,L,Qc,dt)

    In:
    F: NxN Feedback Matrix
    L: FxL Noise effect Matrix
    dt: time step

    Out:
    A: Transition Matrix
    Q: Discrete Process Covariance
    """

    A = expm(F * dt)
    n = F.shape[0]
    Phi = jnp.vstack([jnp.hstack([F, L @ L.T]), jnp.hstack([jnp.zeros((n, n)), -F.T])])
    AB = expm(Phi * dt) @ jnp.vstack([jnp.zeros((n, n)), jnp.eye(n)])
    Q = jnp.linalg.solve(AB[n : (2 * n), :].T, AB[0:n, :].T).T

    return A, Q

if __name__ == "__main__":
    dt = 1
    alpha = 0.1
    F = np.zeros((4,4))
    F[0,2] = 1.0
    F[1,3] = 1.0
    F[2,2] = -alpha
    F[3,3] = -alpha
    L = np.zeros((4,2))
    L[2,0] = 1.0
    L[3,1] = 1.0

    q = 5 
    A, Q = lti_disc(F,q*L, dt=dt)
    print (A)
    print(Q)
