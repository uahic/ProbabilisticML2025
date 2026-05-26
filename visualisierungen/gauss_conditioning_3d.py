import sys
import numpy as np
from scipy.stats import multivariate_normal, norm

from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QSlider, QComboBox
)
from PyQt5.QtCore import Qt

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D


class GaussianConditioningGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Gaussian Conditioning – 2D / 3D")
        self.setGeometry(50, 50, 1400, 700)

        self.init_params()
        self.init_ui()
        self.update_plot()

    def init_params(self):
        self.mu_x = 0.0
        self.mu_y = 0.0
        self.s_xx = 1.0
        self.s_yy = 1.0
        self.s_xy = 0.5
        self.cond_val = 0.0
        self.condition_on = "y"

    def init_ui(self):
        layout = QHBoxLayout(self)

        # ===== Matplotlib =====
        self.fig = Figure(figsize=(11, 6))
        self.canvas = FigureCanvas(self.fig)
        layout.addWidget(self.canvas)

        self.ax2d = self.fig.add_subplot(221)
        self.ax3d = self.fig.add_subplot(223, projection="3d")
        self.ax1d = self.fig.add_subplot(122)

        # ===== Controls =====
        controls = QVBoxLayout()
        layout.addLayout(controls)

        for name, vmin, vmax, init in [
            ("mu_x", -3, 3, self.mu_x),
            ("mu_y", -3, 3, self.mu_y),
            ("s_xx", 0.1, 3, self.s_xx),
            ("s_yy", 0.1, 3, self.s_yy),
            ("s_xy", -2, 2, self.s_xy),
            ("cond_val", -3, 3, self.cond_val),
        ]:
            controls.addLayout(self.create_slider(name, vmin, vmax, init))

        controls.addWidget(QLabel("Condition on:"))
        self.combo = QComboBox()
        self.combo.addItems(["y", "x"])
        self.combo.currentTextChanged.connect(self.change_condition)
        controls.addWidget(self.combo)

        controls.addStretch()

    def create_slider(self, name, vmin, vmax, init):
        layout = QVBoxLayout()
        label = QLabel(f"{name}: {init:.2f}")
        slider = QSlider(Qt.Horizontal)

        slider.setMinimum(0)
        slider.setMaximum(1000)
        slider.setValue(int((init - vmin) / (vmax - vmin) * 1000))

        def scale(val):
            return vmin + (vmax - vmin) * val / 1000

        slider.valueChanged.connect(
            lambda val, n=name, l=label:
            self.slider_changed(n, scale(val), l)
        )

        layout.addWidget(label)
        layout.addWidget(slider)
        return layout

    def slider_changed(self, name, value, label):
        setattr(self, name, value)
        label.setText(f"{name}: {value:.2f}")
        self.update_plot()

    def change_condition(self, text):
        self.condition_on = text
        self.update_plot()

    def update_plot(self):
        self.ax2d.clear()
        self.ax3d.clear()
        self.ax1d.clear()

        Sigma = np.array([
            [self.s_xx, self.s_xy],
            [self.s_xy, self.s_yy]
        ])

        if np.linalg.det(Sigma) <= 0:
            self.canvas.draw()
            return

        mu = np.array([self.mu_x, self.mu_y])

        x = np.linspace(-4, 4, 150)
        y = np.linspace(-4, 4, 150)
        X, Y = np.meshgrid(x, y)
        pos = np.dstack((X, Y))

        rv = multivariate_normal(mu, Sigma)
        Z = rv.pdf(pos)

        # ===== 2D contour =====
        self.ax2d.contourf(X, Y, Z, levels=30, cmap="viridis")
        if self.condition_on == "y":
            self.ax2d.axhline(self.cond_val, color="red", linestyle="--")
        else:
            self.ax2d.axvline(self.cond_val, color="red", linestyle="--")

        self.ax2d.set_title("2D Gaussian")

        # ===== 3D surface =====
        self.ax3d.plot_surface(X, Y, Z, cmap="viridis", linewidth=0, alpha=0.9)
        self.ax3d.set_title("3D Density")
        self.ax3d.set_xlabel("x")
        self.ax3d.set_ylabel("y")

        # ===== Conditioning slice in 3D =====
        if self.condition_on == "y":
            y_slice = np.full_like(x, self.cond_val)
            slice_points = np.column_stack([x, y_slice])
            z_slice = rv.pdf(slice_points)

            # Schnittebene
            self.ax3d.plot(
                x, y_slice, z_slice,
                color="red", linewidth=3, label="p(x, y=y0)"
            )

        else:
            x_slice = np.full_like(y, self.cond_val)
            slice_points = np.column_stack([x_slice, y])
            z_slice = rv.pdf(slice_points)

            self.ax3d.plot(
                x_slice, y, z_slice,
                color="red", linewidth=3, label="p(x=x0, y)"
            )

        self.ax3d.legend()


        # ===== Conditioning =====
        if self.condition_on == "y":
            mu_c = self.mu_x + self.s_xy / self.s_yy * (self.cond_val - self.mu_y)
            var_c = self.s_xx - self.s_xy**2 / self.s_yy
            axis = x
            label = r"$p(x | y=y_0)$"
        else:
            mu_c = self.mu_y + self.s_xy / self.s_xx * (self.cond_val - self.mu_x)
            var_c = self.s_yy - self.s_xy**2 / self.s_xx
            axis = y
            label = r"$p(y | x=x_0)$"

        if var_c > 0:
            self.ax1d.plot(axis, norm.pdf(axis, mu_c, np.sqrt(var_c)), label=label)
            self.ax1d.legend()

        self.ax1d.set_title("Conditioned Distribution")

        self.fig.tight_layout()
        self.canvas.draw()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    gui = GaussianConditioningGUI()
    gui.show()
    sys.exit(app.exec_())

