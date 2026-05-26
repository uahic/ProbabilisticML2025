import sys
import numpy as np
from scipy.stats import multivariate_normal, norm

from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QSlider
)
from PyQt5.QtCore import Qt

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D


# ---------- Hilfsfunktion: Kovarianzellipse ----------

def covariance_ellipse(mu, Sigma, n_std=2.0, n_points=200):
    vals, vecs = np.linalg.eigh(Sigma)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]

    t = np.linspace(0, 2 * np.pi, n_points)
    circle = np.vstack((np.cos(t), np.sin(t)))
    ellipse = vecs @ np.diag(np.sqrt(vals)) @ circle
    ellipse *= n_std
    ellipse[0] += mu[0]
    ellipse[1] += mu[1]
    return ellipse


# ---------- GUI ----------

class GaussianConditioningGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Gaussian Conditioning – Correct & Stable")
        self.setGeometry(50, 50, 1650, 850)

        # feste Weltkoordinaten
        self.xlim = (-4, 4)
        self.ylim = (-4, 4)

        self.init_params()
        self.init_ui()
        self.update_plot()

    def init_params(self):
        self.mu_x = 0.0
        self.mu_y = 0.0
        self.s_xx = 1.0
        self.s_yy = 1.0
        self.s_xy = 0.6

        # Konditionierungsgerade a1 x + a2 y = z0
        self.a1 = 1.0
        self.a2 = 0.0
        self.z0 = 0.0

    def init_ui(self):
        layout = QHBoxLayout(self)

        # ===== Matplotlib =====
        self.fig = Figure(figsize=(12, 6))
        self.canvas = FigureCanvas(self.fig)
        layout.addWidget(self.canvas)

        self.ax2d = self.fig.add_subplot(221)
        self.ax3d = self.fig.add_subplot(223, projection="3d")
        self.ax1d = self.fig.add_subplot(122)

        # ===== Controls =====
        controls = QVBoxLayout()
        layout.addLayout(controls)

        sliders = [
            ("mu_x", -3, 3),
            ("mu_y", -3, 3),
            ("s_xx", 0.2, 3),
            ("s_yy", 0.2, 3),
            ("s_xy", -2, 2),
            ("a1", -2, 2),
            ("a2", -2, 2),
            ("z0", -3, 3),
        ]

        for name, vmin, vmax in sliders:
            controls.addLayout(self.make_slider(name, vmin, vmax))

        self.model_label = QLabel()
        self.model_label.setAlignment(Qt.AlignTop)
        self.model_label.setStyleSheet(
            "QLabel { font-family: Courier New; font-size: 11px; padding: 10px;"
            "background-color: #f4f4f4; border: 1px solid #ccc; }"
        )

        controls.addWidget(self.model_label)
        controls.addStretch()

    def make_slider(self, name, vmin, vmax):
        layout = QVBoxLayout()
        label = QLabel(f"{name}: {getattr(self, name):.2f}")
        slider = QSlider(Qt.Horizontal)
        slider.setMinimum(0)
        slider.setMaximum(1000)

        def scale(v):
            return vmin + (vmax - vmin) * v / 1000

        slider.setValue(int((getattr(self, name) - vmin) / (vmax - vmin) * 1000))
        slider.valueChanged.connect(
            lambda v, n=name, l=label: self.update_param(n, scale(v), l)
        )

        layout.addWidget(label)
        layout.addWidget(slider)
        return layout

    def update_param(self, name, value, label):
        setattr(self, name, value)
        label.setText(f"{name}: {value:.2f}")
        self.update_plot()

    def update_plot(self):
        self.ax2d.clear()
        self.ax3d.clear()
        self.ax1d.clear()

        # ===== Prior =====
        mu = np.array([self.mu_x, self.mu_y])
        Sigma = np.array([[self.s_xx, self.s_xy],
                          [self.s_xy, self.s_yy]])

        if np.linalg.det(Sigma) <= 0:
            self.canvas.draw()
            return

        x = np.linspace(*self.xlim, 300)
        y = np.linspace(*self.ylim, 300)
        X, Y = np.meshgrid(x, y)
        rv = multivariate_normal(mu, Sigma, allow_singular=True)
        Z = rv.pdf(np.dstack((X, Y)))

        # ===== 2D Prior-Dichte (stabil) =====
        self.ax2d.set_autoscale_on(False)
        self.ax2d.set_xlim(self.xlim)
        self.ax2d.set_ylim(self.ylim)

        self.ax2d.contourf(X, Y, Z, levels=30, cmap="viridis")

        # Konditionierungsgerade
        if abs(self.a2) > 1e-6:
            y_line = (self.z0 - self.a1 * x) / self.a2
            self.ax2d.plot(x, y_line, "r--", lw=2)
        else:
            self.ax2d.axvline(self.z0 / self.a1, color="red", ls="--")

        # Prior-Ellipse
        self.ax2d.plot(*covariance_ellipse(mu, Sigma), "w", lw=2, label="prior")

        # ===== Allgemeine lineare Konditionierung =====
        A = np.array([[self.a1, self.a2]])
        S = A @ Sigma @ A.T
        K = Sigma @ A.T / S[0, 0]

        mu_c = mu + (K @ (self.z0 - A @ mu)).flatten()
        Sigma_c = Sigma - K @ A @ Sigma

        # Posterior-Ellipse (geometrisch!)
        self.ax2d.plot(*covariance_ellipse(mu_c, Sigma_c), "r", lw=2,
                       label="posterior covariance")

        self.ax2d.scatter(*mu, color="white", s=30)
        self.ax2d.scatter(*mu_c, color="red", s=30)

        # ===== Bedingte Dichte (korrekt: 1D auf der Geraden) =====
        t = np.linspace(-6, 6, 500)

        if abs(self.a2) > 1e-6:
            xs = t
            ys = (self.z0 - self.a1 * xs) / self.a2
        else:
            xs = np.full_like(t, self.z0 / self.a1)
            ys = t

        pts = np.column_stack([xs, ys])
        z_vals = multivariate_normal(mu_c, Sigma_c, allow_singular=True).pdf(pts)
        z_vals = z_vals / z_vals.max()  # normiert für Farbe

        self.ax2d.scatter(
            xs, ys,
            c=z_vals,
            cmap="Reds",
            s=6,
            alpha=0.9,
            label="conditioned density"
        )

        self.ax2d.legend()
        self.ax2d.set_title("2D Gaussian (fixed scale, correct conditioning)")

        # ===== 3D =====
        self.ax3d.plot_surface(X, Y, Z, cmap="viridis", alpha=0.85, linewidth=0)

        self.ax3d.plot(xs, ys, z_vals * Z.max(), "r", lw=3)
        self.ax3d.set_title("3D density + conditioning slice")

        # ===== 1D bedingte Dichte =====
        proj_var = (A @ Sigma @ A.T)[0, 0]
        proj_mu = (A @ mu)[0]

        self.ax1d.plot(
            t,
            norm.pdf(t, proj_mu, np.sqrt(proj_var)),
            lw=2
        )
        self.ax1d.set_title("p(z = A x)")

        # ===== Mathematisches Modell =====
        self.model_label.setText(
            "Model\n"
            "-----------------------------\n"
            "x ~ N(μ, Σ)\n\n"
            f"μ = [{self.mu_x:.2f}, {self.mu_y:.2f}]ᵀ\n"
            f"Σ = [[{self.s_xx:.2f}, {self.s_xy:.2f}],\n"
            f"     [{self.s_xy:.2f}, {self.s_yy:.2f}]]\n\n"
            "Conditioning:\n"
            f"z = [{self.a1:.2f}, {self.a2:.2f}]·[x,y]ᵀ = {self.z0:.2f}\n\n"
            "Update:\n"
            "μₙ = μ + ΣAᵀ(AΣAᵀ)⁻¹(z − Aμ)\n"
            "Σₙ = Σ − ΣAᵀ(AΣAᵀ)⁻¹AΣ\n"
        )

        self.fig.tight_layout()
        self.canvas.draw()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    gui = GaussianConditioningGUI()
    gui.show()
    sys.exit(app.exec_())

