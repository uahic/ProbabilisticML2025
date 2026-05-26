
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from scipy.special import expit  # Sigmoid

# ------------------------
# 1. Trainingsdaten
# ------------------------
X = np.array([[1.0, 2.0],
              [2.0, 1.0]])
Y = np.array([1, -1])

# ------------------------
# 2. ReLU Feature Mapping
# ------------------------
W = np.eye(2)   # Identity für Einfachheit
b = np.zeros(2)

def relu(z):
    return np.maximum(0, z)

def phi(x):
    return relu(W @ x + b)

# ------------------------
# 3. Likelihood
# ------------------------
def likelihood(theta):
    # theta: 1D vector
    L = 1.0
    for x, y in zip(X, Y):
        z = y * np.dot(phi(x), theta)
        L *= expit(z)
    return L

# ------------------------
# 4. Prior
# ------------------------
m = np.zeros(2)           # Mittelwert
V = np.eye(2) * 2.0       # Kovarianz
prior = multivariate_normal(mean=m, cov=V)

def prior_pdf(theta):
    return prior.pdf(theta)

# ------------------------
# 5. Posterior (unnormalized)
# ------------------------
def posterior(theta):
    return likelihood(theta) * prior_pdf(theta)

# ------------------------
# 6. Grid für Plot
# ------------------------
theta1 = np.linspace(-5, 5, 200)
theta2 = np.linspace(-5, 5, 200)
T1, T2 = np.meshgrid(theta1, theta2)

L_grid = np.zeros_like(T1)
P_grid = np.zeros_like(T1)
Post_grid = np.zeros_like(T1)

for i in range(T1.shape[0]):
    for j in range(T1.shape[1]):
        th = np.array([T1[i,j], T2[i,j]])
        L_grid[i,j] = likelihood(th)
        P_grid[i,j] = prior_pdf(th)
        Post_grid[i,j] = posterior(th)

# ------------------------
# 7. Plot Likelihood, Prior, Posterior
# ------------------------
fig, axs = plt.subplots(1,3, figsize=(18,5))

# Likelihood
cp0 = axs[0].contourf(T1, T2, L_grid, levels=50, cmap='viridis')
fig.colorbar(cp0, ax=axs[0])
axs[0].set_title("Likelihood")
axs[0].set_xlabel(r"$\theta_1$")
axs[0].set_ylabel(r"$\theta_2$")

# Prior
cp1 = axs[1].contourf(T1, T2, P_grid, levels=50, cmap='plasma')
fig.colorbar(cp1, ax=axs[1])
axs[1].set_title("Prior")
axs[1].set_xlabel(r"$\theta_1$")
axs[1].set_ylabel(r"$\theta_2$")

# Posterior
cp2 = axs[2].contourf(T1, T2, Post_grid, levels=50, cmap='inferno')
fig.colorbar(cp2, ax=axs[2])
axs[2].set_title("Posterior (unnormalized)")
axs[2].set_xlabel(r"$\theta_1$")
axs[2].set_ylabel(r"$\theta_2$")

plt.tight_layout()
plt.show()
