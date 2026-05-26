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


class GaussianConditioningGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("2D Gaussian Conditioning")
        self.setGeometry(100, 100, 1100, 500)

        self.init_params()
        self.init_ui()
        self.update_plot()

    def init_params(self):
        self.mu_x = 0.0
        self.mu_y = 0.0
        self.s_xx = 1.0
        self.s_yy = 1.0
        self.s_xy = 0.5
        self.y0 = 0.0

    def init_ui(self):
        layout = QHBoxLayout(self)

        # ===== Matplotlib Figure =====
        self.fig = Figure(figsize=(10, 4))
        self.canvas = FigureCanvas(self.fig)
        layout.addWidget(self.canvas)

        self.ax2d = self.fig.add_subplot(121)
        self.ax1d = self.fig.add_subplot(122)

        # ===== Controls =====
        controls = QVBoxLayout()
        layout.addLayout(controls)

        self.sliders = {}
        for name, vmin, vmax, init in [
            ("mu_x", -3, 3, self.mu_x),
            ("mu_y", -3, 3, self.mu_y),
            ("s_xx", 0.1, 3, self.s_xx),
            ("s_yy", 0.1, 3, self.s_yy),
            ("s_xy", -2, 2, self.s_xy),
            ("y0", -3, 3, self.y0),
        ]:
            controls.addLayout(self.create_slider(name, vmin, vmax, init))

        controls.addStretch()

    def create_slider(self, name, vmin, vmax, init):
        layout = QVBoxLayout()
        label = QLabel(f"{name}: {init:.2f}")
        slider = QSlider(Qt.Horizontal)

        slider.setMinimum(0)
        slider.setMaximum(1000)

        def scale(val):
            return vmin + (vmax - vmin) * val / 1000

        slider.setValue(int((init - vmin) / (vmax - vmin) * 1000))

        slider.valueChanged.connect(
            lambda val, n=name, l=label:
            self.slider_changed(n, scale(val), l)
        )

        self.sliders[name] = slider
        layout.addWidget(label)
        layout.addWidget(slider)
        return layout

    def slider_changed(self, name, value, label):
        setattr(self, name, value)
        label.setText(f"{name}: {value:.2f}")
        self.update_plot()

    def update_plot(self):
        self.ax2d.clear()
        self.ax1d.clear()

        # Covariance matrix (ensure positive definite)
        Sigma = np.array([
            [self.s_xx, self.s_xy],
            [self.s_xy, self.s_yy]
        ])

        if np.linalg.det(Sigma) <= 0:
            self.canvas.draw()
            return

        mu = np.array([self.mu_x, self.mu_y])

        # ===== 2D Gaussian =====
        x = np.linspace(-4, 4, 300)
        y = np.linspace(-4, 4, 300)
        X, Y = np.meshgrid(x, y)
        pos = np.dstack((X, Y))

        rv = multivariate_normal(mu, Sigma)
        Z = rv.pdf(pos)

        self.ax2d.contourf(X, Y, Z, levels=30, cmap="viridis")
        self.ax2d.axhline(self.y0, color="red", linestyle="--")
        self.ax2d.set_title("2D Gaussian")
        self.ax2d.set_xlabel("x")
        self.ax2d.set_ylabel("y")

        # ===== Conditioning =====
        mu_x, mu_y = mu
        Sigma_xx = Sigma[0, 0]
        Sigma_xy = Sigma[0, 1]
        Sigma_yy = Sigma[1, 1]

        mu_cond = mu_x + Sigma_xy / Sigma_yy * (self.y0 - mu_y)
        var_cond = Sigma_xx - Sigma_xy**2 / Sigma_yy

        if var_cond > 0:
            px = norm.pdf(x, mu_cond, np.sqrt(var_cond))
            self.ax1d.plot(x, px, label=r"$p(x | y=y_0)$")
            self.ax1d.legend()

        self.ax1d.set_title("Conditioned Distribution")
        self.ax1d.set_xlabel("x")
        self.ax1d.set_ylabel("Density")

        self.fig.tight_layout()
        self.canvas.draw()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    gui = GaussianConditioningGUI()
    gui.show()
    sys.exit(app.exec_())

