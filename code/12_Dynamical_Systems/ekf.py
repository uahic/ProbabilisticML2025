from jax import numpy as jnp
from jax import jacrev

def ExtendedKalmanFilter(f: callable, h: callable, Q, R):
    """
    Instead of global linear transforms this filter uses linearizations of non-linear functions f and h
    f: general monotonic function
    h: general monotonic function
    Q: Process Noise
    R: Observation Noise
    """

    def predict(m, P):
        m_ = f(m)  #  Instead of A*m
        F = jacrev(f)(m)  # Compute jacobian of f evaluated at m to linearize the system dynamic at m
        P_ = F @ P @ F.T + Q  # linear-transform the uncertainty
        return m_, P_

    def update(m_, P_, z):
        z_ = h(m_)
        H = jacrev(h)(m_)
        S = H @ P_ @ H.T + R
        K = jnp.linalg.solve(S, H @ P_).T
        m = m_ + K @ (z - z_)
        P = P_ - K @ S @ K.T
        return m, P

    return predict, update