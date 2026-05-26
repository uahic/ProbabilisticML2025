import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QSlider, QLabel, QComboBox, QCheckBox, QHBoxLayout

# ----- Funktionen -----
def p_x(x, mu, sigma):
    # Normalverteilung (Gaussian)
    return (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

def p_y(y, mu, sigma, use_jacobian, u_function):
    # Transformation u(x) -> y = u(x), und x = inverse(u)
    x = u_function(y)
    dx_dy = np.abs(np.gradient(u_function(y), y))  # Der Jacobi-Faktor
    base = p_x(x, mu, sigma)
    if use_jacobian:
        return base * dx_dy  # Apply Jacobian factor
    else:
        return base

def u_linear(x):
    return 0.5 * x

def u_squared(x):
    return x**2

def u_sin(x):
    return np.sin(x)

# ----- GUI-Layout und Interaktive Komponenten -----
class PlotWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Interaktive Wahrscheinlichkeitsverteilung")
        self.setGeometry(100, 100, 800, 600)

        # Initiale Werte für Mittelwert, Varianz, Jacobi-Faktor und Transformation
        self.mu = 0.75
        self.sigma = 0.12
        self.use_jacobian = True
        self.u_function = u_linear  # Standard: lineare Funktion

        # Erstelle das Layout
        self.layout = QVBoxLayout()

        # Matplotlib Figure und Canvas
        self.fig, self.axes = plt.subplots(2, 2, figsize=(8, 8), gridspec_kw={'width_ratios': [1, 4], 'height_ratios': [4, 1]})
        self.canvas = FigureCanvas(self.fig)
        self.layout.addWidget(self.canvas)

        # Schieberegler für Mittelwert (mu)
        self.slider_mu = QSlider()
        self.slider_mu.setOrientation(1)  # 1 = horizontal
        self.slider_mu.setMinimum(0)
        self.slider_mu.setMaximum(100)
        self.slider_mu.setValue(int(self.mu * 100))
        self.slider_mu.valueChanged.connect(self.update_plot)
        self.layout.addWidget(QLabel("Mittelwert (mu)"))
        self.layout.addWidget(self.slider_mu)

        # Schieberegler für Varianz (sigma)
        self.slider_sigma = QSlider()
        self.slider_sigma.setOrientation(1)
        self.slider_sigma.setMinimum(1)
        self.slider_sigma.setMaximum(50)
        self.slider_sigma.setValue(int(self.sigma * 100))
        self.slider_sigma.valueChanged.connect(self.update_plot)
        self.layout.addWidget(QLabel("Varianz (sigma)"))
        self.layout.addWidget(self.slider_sigma)

        # Dropdown-Menü für Funktionswahl
        self.function_selector = QComboBox()
        self.function_selector.addItem("Lineare Funktion (u(x) = 0.5x)")
        self.function_selector.addItem("Quadratische Funktion (u(x) = x^2)")
        self.function_selector.addItem("Sinus-Funktion (u(x) = sin(x))")
        self.function_selector.currentIndexChanged.connect(self.update_function)
        self.layout.addWidget(QLabel("Wähle eine Funktion"))
        self.layout.addWidget(self.function_selector)

        # Checkbox für Jacobi-Faktor
        self.jacobi_checkbox = QCheckBox("Jacobi Faktor an")
        self.jacobi_checkbox.setChecked(True)
        self.jacobi_checkbox.stateChanged.connect(self.update_jacobi)
        self.layout.addWidget(self.jacobi_checkbox)

        # Container für alle Layouts
        container = QWidget()
        container.setLayout(self.layout)
        self.setCentralWidget(container)

        # Initiale Plot-Darstellung
        self.update_plot()

        # Maus-Klick-Event für Interaktivität
        self.cid = self.fig.canvas.mpl_connect('button_press_event', self.on_click)

    def update_function(self):
        """Aktualisiere die gewählte Funktion für die Transformation"""
        index = self.function_selector.currentIndex()
        if index == 0:
            self.u_function = u_linear
        elif index == 1:
            self.u_function = u_squared
        elif index == 2:
            self.u_function = u_sin
        self.update_plot()

    def update_jacobi(self):
        """Aktualisiere den Jacobi-Faktor"""
        self.use_jacobian = self.jacobi_checkbox.isChecked()
        self.update_plot()

    def update_plot(self):
        """Aktualisiere den Plot basierend auf den Slider-Werten"""
        self.mu = self.slider_mu.value() / 100
        self.sigma = self.slider_sigma.value() / 100

        xs = np.linspace(0, 1, 400)
        ys = np.linspace(0, 1, 400)

        # Berechnung von p(x) und p(y)
        ax_main = self.axes[0, 1]
        ax_px = self.axes[1, 1]
        ax_py = self.axes[0, 0]

        # Hauptplot: y = u(x)
        ax_main.clear()
        ax_main.plot(xs, self.u_function(xs), color="black", lw=1.5, label="u(x)")
        ax_main.set_xlim(0, 1)
        ax_main.set_ylim(0, 1)
        ax_main.set_xlabel("x")
        ax_main.set_ylabel("y")
        ax_main.set_title("Transformation u(x)")
        ax_main.grid(True)

        # Rand unten: p(x)
        ax_px.clear()
        ax_px.plot(xs, p_x(xs, self.mu, self.sigma), color="tab:blue")
        ax_px.set_xlim(0, 1)
        ax_px.set_xlabel("x")
        ax_px.set_ylabel("p(x)")
        ax_px.grid(True)

        # Rand links: p(y) vertikal, horizontal gespiegelt (nach links)
        ax_py.clear()
        py_vals = p_y(ys, self.mu, self.sigma, self.use_jacobian, self.u_function)
        ax_py.plot(py_vals, ys, color="tab:orange")
        ax_py.set_ylim(0, 1)
        ax_py.set_xlabel("")  # Platz sparen
        ax_py.set_ylabel("y")
        ax_py.grid(True)

        # Zeichne transformierte Fläche und vertikale Linien für Jacobi-Faktor Visualisierung
        self.highlight_area(ax_px, ax_py)

        self.canvas.draw()

    def highlight_area(self, ax_px, ax_py):
        """Zeichne nur die transformierte Fläche unter den Kurven p(x) und p(y) für den Jacobi-Faktor"""
        # Wähle einen x0, z.B. in der Mitte
        x0 = 0.5
        dx = 0.1

        # Berechne den entsprechenden y0 und die Fläche um y0
        y0 = self.u_function(x0)
        dy = 0.05  # Die Breite der Fläche in der y-Achse, die wir einfärben möchten

        # Berechne den Bereich von x, der der Fläche bei x0 entspricht
        x_vals = np.linspace(x0 - dx / 2, x0 + dx / 2, 10)
        y_vals = np.linspace(y0 - dy / 2, y0 + dy / 2, 10)

        p_x_vals = p_x(x_vals, self.mu, self.sigma)
        p_y_vals = p_y(y_vals, self.mu, self.sigma, self.use_jacobian, self.u_function)

        # Färbe die transformierte Fläche unter den Kurven
        ax_px.fill_between(x_vals, 0, p_x_vals, color="blue", alpha=0.2)
        ax_py.fill_betweenx(y_vals, 0, p_y_vals, color="orange", alpha=0.2)

        # Vertikale Linie bei x0 und y0
        ax_px.axvline(x0, color="red", lw=1.5, linestyle="--")
        ax_py.axvline(y0, color="red", lw=1.5, linestyle="--")

    def on_click(self, event):
        """Reagiere auf Maus-Klicks und zeige nur die transformierte Fläche"""
        if event.inaxes == self.axes[1, 1]:  # Prüfe, ob der Klick im px-Plot war
            x0 = event.xdata
            self.highlight_area(self.axes[1, 1], self.axes[0, 0])
            self.update_plot()

# ----- Main Loop -----
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = PlotWindow()
    window.show()
    sys.exit(app.exec_())

