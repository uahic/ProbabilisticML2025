# Fur f(x) = [cos(x), x**2]
# Als parametrische Kurve  (Cov ist: J*[sigma]*J.T)
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib.patches import Ellipse

# ---------------- Funktion und Jacobi ----------------
def f(x):
    return np.array([np.cos(x), x**2])

def jacobian(x):
    return np.array([
        [-np.sin(x)],
        [2 * x]
    ])  # (2x1)

# ---------------- Unsicherheitsellipse ----------------
def covariance_ellipse(mean, cov, n_std=2.0, **kwargs):
    vals, vecs = np.linalg.eigh(cov)

    # Sortiere Eigenwerte absteigend
    order = vals.argsort()[::-1]
    vals = vals[order]
    vecs = vecs[:, order]

    angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
    width, height = 2 * n_std * np.sqrt(np.maximum(vals, 0))

    return Ellipse(mean, width, height, angle=angle, **kwargs)

# ---------------- Parameter ----------------
sigma_x = 0.2  # Standardabweichung von x

x_vals = np.linspace(-3.5, 3.5, 1000)
curve = np.array([f(x) for x in x_vals])

# ---------------- Plot ----------------
fig, ax = plt.subplots(figsize=(7, 7))
plt.subplots_adjust(bottom=0.25)

ax.plot(curve[:, 0], curve[:, 1], label=r'$f(x) = (\cos x, x^2)$')

# Punkt als Scatter (robuster als plot)
point = ax.scatter([], [], color='red', zorder=5, label='m = f(x)')
ellipse = None

ax.set_xlabel(r'$\cos(x)$')
ax.set_ylabel(r'$x^2$')
ax.set_title('Punkt auf der Kurve mit EKF-Unsicherheitsellipse')
ax.grid(True)
ax.axis('equal')
ax.legend()

# ---------------- Slider ----------------
ax_slider = plt.axes([0.2, 0.1, 0.6, 0.03])
slider = Slider(ax_slider, 'x', -3.5, 3.5, valinit=0.0)

# ---------------- Update-Funktion ----------------
def update(val):
    global ellipse

    x = slider.val
    m = f(x)

    # Punkt aktualisieren
    point.set_offsets([m])

    # EKF-Kovarianz
    J = jacobian(x)           # (2x1)
    P = sigma_x**2 * (J @ J.T)  # (2x2), korrekt

    # Alte Ellipse entfernen
    if ellipse is not None:
        ellipse.remove()

    # Neue Ellipse
    ellipse = covariance_ellipse(
        mean=m,
        cov=P,
        n_std=2,
        edgecolor='red',
        facecolor='none',
        linewidth=2
    )
    ax.add_patch(ellipse)

    fig.canvas.draw_idle()

slider.on_changed(update)

# Initialer Zustand
update(0.0)

plt.show()
