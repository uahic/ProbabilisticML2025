import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib.patches import Ellipse

# ---------------- 2D-Systemdynamik ----------------
def f(xy):
    x, y = xy
    return np.array([x + 0.5*np.sin(y), y + 0.5*np.sin(x)])

def jacobian(xy):
    x, y = xy
    return np.array([
        [1, 0.5*np.cos(y)],
        [0.5*np.cos(x), 1]
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
steps = 5

# ---------------- Plot vorbereiten ----------------
fig, ax = plt.subplots(figsize=(8,8))
plt.subplots_adjust(bottom=0.25)

ax.set_xlim(-3, 3)
ax.set_ylim(-3, 3)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('2D-Systemdynamik mit EKF-Ellipsen, Trajektorie, Vektorfeld und Konturfarben')
ax.grid(True)
ax.set_aspect('equal')

# ---------------- Höhenkontur ----------------
X = np.linspace(-3, 3, 200)
Y = np.linspace(-3, 3, 200)
XX, YY = np.meshgrid(X, Y)
ZZ = np.sqrt((0.5*np.sin(YY))**2 + (0.5*np.sin(XX))**2)
contour = ax.contourf(XX, YY, ZZ, levels=30, cmap='viridis', alpha=0.6)
cbar = plt.colorbar(contour, ax=ax)
cbar.set_label('||f(x,y)-[x,y]||')

# ---------------- Vektorfeld ----------------
XX_v, YY_v = np.meshgrid(np.linspace(-3, 3, 20), np.linspace(-3, 3, 20))
UU, VV = np.zeros_like(XX_v), np.zeros_like(YY_v)
for i in range(XX_v.shape[0]):
    for j in range(XX_v.shape[1]):
        vec = f([XX_v[i,j], YY_v[i,j]])
        UU[i,j] = vec[0] - XX_v[i,j]
        VV[i,j] = vec[1] - YY_v[i,j]

ax.quiver(XX_v, YY_v, UU, VV, color='gray', alpha=0.6)

# ---------------- Punkte, Ellipsen, Trajektorie ----------------
state_points = []
state_ellipses = []
traj_line, = ax.plot([], [], 'k-o', label='Trajektorie')

# ---------------- Slider ----------------
ax_slider_x = plt.axes([0.2, 0.1, 0.6, 0.03])
slider_x = Slider(ax_slider_x, 'x0', -2.0, 2.0, valinit=0.5)

ax_slider_y = plt.axes([0.2, 0.05, 0.6, 0.03])
slider_y = Slider(ax_slider_y, 'y0', -2.0, 2.0, valinit=0.5)

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

    xy = np.array([x0, y0])
    P = P0.copy()
    traj = [xy.copy()]

    for _ in range(steps):
        # Zustandsellipse
        e = covariance_ellipse(xy, P, n_std=2, edgecolor='blue', facecolor='none', linewidth=2)
        ax.add_patch(e)
        state_ellipses.append(e)

        # Punkt
        p = ax.scatter([xy[0]], [xy[1]], color='red', zorder=5)
        state_points.append(p)

        # EKF-Prop
        F = jacobian(xy)
        P = F @ P @ F.T + Q
        xy = f(xy)
        traj.append(xy.copy())

    traj = np.array(traj)
    traj_line.set_data(traj[:,0], traj[:,1])

    fig.canvas.draw_idle()

slider_x.on_changed(update)
slider_y.on_changed(update)

update(0.0)
ax.legend()
plt.show()
