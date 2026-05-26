import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib.patches import Ellipse

# ---------------- 2D-Systemdynamik ----------------
def f(xy):
    x, y = xy
    return np.array([x**2 - y, x + y**2])

# Jacobi der Dynamik
def jacobian(xy):
    x, y = xy
    return np.array([
        [2*x, -1],
        [1, 2*y]
    ])

# ---------------- Unsicherheitsellipse ----------------
def covariance_ellipse(mean, cov, n_std=2.0, **kwargs):
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals = vals[order]
    vecs = vecs[:, order]

    angle = np.degrees(np.arctan2(vecs[1,0], vecs[0,0]))
    width, height = 2 * n_std * np.sqrt(np.maximum(vals,0))
    return Ellipse(mean, width, height, angle=angle, **kwargs)

# ---------------- Parameter ----------------
sigma_state = 0.2
P0 = np.array([[sigma_state**2, 0],
               [0, sigma_state**2]])
Q = np.array([[0.01, 0],
              [0, 0.01]])
steps = 5  # wie viele Iterationen

# ---------------- Plot vorbereiten ----------------
fig, ax = plt.subplots(figsize=(8,8))
plt.subplots_adjust(bottom=0.25)

ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('2D-Systemdynamik mit EKF-Ellipsen und Flussfeld')
ax.grid(True)
ax.set_aspect('equal')

# ---------------- Vektorfeld (f) ----------------
X = np.linspace(-2, 2, 20)
Y = np.linspace(-2, 2, 20)
XX, YY = np.meshgrid(X, Y)
UU = np.zeros_like(XX)
VV = np.zeros_like(YY)
for i in range(XX.shape[0]):
    for j in range(XX.shape[1]):
        vec = f([XX[i,j], YY[i,j]])
        UU[i,j], VV[i,j] = vec[0], vec[1]

ax.quiver(XX, YY, UU, VV, color='gray', alpha=0.5)

# ---------------- Punkte, Ellipsen, Trajektorie ----------------
state_points = []
state_ellipses = []
traj_line, = ax.plot([], [], 'k-o', label='Trajektorie')

# ---------------- Slider ----------------
ax_slider_x = plt.axes([0.2, 0.1, 0.6, 0.03])
slider_x = Slider(ax_slider_x, 'x0', -1.0, 1.0, valinit=0.2)

ax_slider_y = plt.axes([0.2, 0.05, 0.6, 0.03])
slider_y = Slider(ax_slider_y, 'y0', -1.0, 1.0, valinit=0.2)

# ---------------- Update ----------------
def update(val):
    global state_points, state_ellipses
    x0 = slider_x.val
    y0 = slider_y.val

    # Alte Punkte/Ellipsen entfernen
    for p in state_points:
        p.remove()
    for e in state_ellipses:
        e.remove()
    state_points = []
    state_ellipses = []

    # Initialzustand
    xy = np.array([x0, y0])
    P = P0.copy()
    traj = [xy.copy()]

    # Simulation über mehrere Schritte
    for _ in range(steps):
        # Ellipse zeichnen
        e = covariance_ellipse(xy, P, n_std=2, edgecolor='blue', facecolor='none', linewidth=2)
        ax.add_patch(e)
        state_ellipses.append(e)

        # Punkt zeichnen
        p = ax.scatter([xy[0]], [xy[1]], color='red', zorder=5)
        state_points.append(p)

        # EKF-Prop: x_k+1 = f(x_k), P_k+1 = F P F^T + Q
        F = jacobian(xy)
        P = F @ P @ F.T + Q
        xy = f(xy)
        traj.append(xy.copy())

    # Trajektorie aktualisieren
    traj = np.array(traj)
    traj_line.set_data(traj[:,0], traj[:,1])

    fig.canvas.draw_idle()

slider_x.on_changed(update)
slider_y.on_changed(update)

update(0.0)
ax.legend()
plt.show()
