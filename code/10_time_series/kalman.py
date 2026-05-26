from jax import numpy as jnp


def KalmanFilter(A, Q, H, R):
    """
    A Factory class

    Inputs:
    A: Transition model
    Q: Process Noise
    H: Observation Model
    R: Observation Noise
    """

    def predict(m, P):
        """
        Linear Transformation N(xt; Am; APA.T +Q)
        to update state p(xt | xt-1)

        m: old mean value (of the state)
        P: old uncertainty (cov matrix)
        """
        m_ = A @ m
        P_ = A @ P @ A.T + Q
        return m_, P_

    def update(m_, P_, z):
        """
        Bayesian update step to get from p(xt | yt-1) -> p(xt | yt)
        z: observation
        m_: predicted mean
        P_: predicted uncertainty
        """
        S = H @ P_ @ H.T + R  # transform predicted uncertainty into observation space
        K = jnp.linalg.solve(S.T, H @ P_).T  # Kalman factor
        m = m_ + K @ (z - H @ m_)  # update old mean, K weights the residual
        P = P_ - K @ H @ P_  # Reduction of uncertainty
        return m, P

    def smooth(ms, Ps, m, P):
        G = jnp.linalg.solve(P, A @ Ps).T
        ms = m + G @ (ms - m)
        Ps = P + G @ (Ps - P) @ G.T
        return ms, Ps

    return predict, update, smooth



