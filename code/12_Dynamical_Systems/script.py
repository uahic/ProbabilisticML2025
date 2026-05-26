from jax import numpy as jnp
from jax import random
from jax import jacrev

## Extended Kalman filter, Covid example [Lecture 12, Minute 25:00]
from jax.nn import sigmoid
from lti import lti_disc
from ekf import ExtendedKalmanFilter

# Stochastic differential equation (SDE)
# dI(t) = F*dI(t)dt + Ldw(t)dt
A, Q = lti_disc(F=jnp.zeros((1, 1)), L=jnp.eye(1) * 0.2, dt=1)
ALPHA = 1e-6


def h(x):
    """
    Transformation function between two random variables
    (1+ALPHA) to make sure super small values of x dont run into numerical issues
    """
    return (1 + ALPHA) * sigmoid(x) - ALPHA / 2


# For fancy plots
def h_inv(y):
    z = (y + ALPHA / 2) / (1 + ALPHA)
    return jnp.log(z) - jnp.log(1 - z)


# f can also be non-linear (which gets linearized by the EKF)
f = lambda x: A @ x
R = 1e-6**2 * jnp.eye(1)

predict, update = ExtendedKalmanFilter(f, h, Q, R)

# TODO plot

# Now with derivatives of the state [Lecture 12, Minute 30:00]

# X(t) = [
#     I(t),
#     I^{*}(t),
#     I^{**}(t)
# ]

# dx(t) = [
#     dI(t),
#     dI^{*}(t),
#     dI^{**}(t),
# ] = 
# F = [ [0, 1, 0],
#       [0, 0 ,1],
#       [0, 0 ,0]
#      ] * dt +

# In L, we have to set the first values to zero, otherwise the first two entries of dx(t) are not derivates anymore
# L = [
#     [0],
#     [0],
#     [sigma]
# ] * dt

num_derivatives = 3
sigma = 4e-6
F = jnp.zeros(1 + num_derivatives, 1 + num_derivatives)
L = jnp.zeros(1 + num_derivatives, 1)
F = F.at[i, :, i, :].set(jnp.diag(jnp.ones(num_derivatives - 1), k=1))
L = L.at[i, 2, i, 2].set(L[i])
A, Q = lti_disc(F, L, dt=1)

# Prinzip:
# - Was man modellieren will ist als SDE definiert (z.b. Infektionsraten)
# - Dies wird in Matritzen fur Gauss(prozesse) umgewandelt
# - Diese wird Matrix A wird in f reingesteckt (als A*x)
# - Im EKF wird f (und damit A) differenziert (zur Linearisierung, obwohl streng genommen isses schon linear)

def h_state(X):
    return h(X[(0,),])  # weird stuff so input stays array, X[(0,)] refers to the first entry of x(t) and we take the sigmoid (RV->RV trafo) of it


f = lambda x: A @ x


predict, update = ExtendedKalmanFilter(f, h_state, Q, R)


# ODEs
def h_ODE(X):
    J = jacrev(h,)(X[(0,3,6),])
    Xdot = jnp.array([J[0,0] * X[1], J[1,1]*X[4], J[2,2]*X[7]])
    ODEinputs = [h(X[0]), h(X[3]), h(X[6])] # SIR inputs
    return Xdot - ode(*ODEinputs)