%matplotlib notebook
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import CheckButtons
from matplotlib.gridspec import GridSpec
from ipywidgets import interactive, FloatSlider, ToggleButton
import matplotlib.pyplot as plt

# ----- Verteilungen -----
def p_x(x, mu, sigma):
    # Normalverteilung (Gaussian)
    return (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

def p_y(y, mu, sigma, use_jacobian):
    # Transformation y = 0.5 * x  ->  x = 2 * y
    x = 2 * y
    base = p_x(x, mu, sigma)
    return 2 * base if use_jacobian else base

# Domains in [0, 1]
xs = np.linspace(0, 1, 400)
ys = np.linspace(0, 1, 400)

# ----- Funktion für das Plotten -----
def update_plot(mu, sigma, use_jacobian):
    fig, axes = plt.subplots(2, 2, figsize=(8, 8), gridspec_kw={'width_ratios': [1, 4], 'height_ratios': [4, 1]})
    ax_py = axes[0, 0]  # links oben: vertikale p(y)
    ax_main = axes[0, 1]  # rechts oben: Hauptplot y = 0.5 x
    ax_px = axes[1, 1]  # rechts unten: horizontale p(x)
    ax_check = axes[1, 0]  # kleine Achse links unten für Controls (Checkbox)
    
    ax_check.axis("off")  # leeres Feld
    
    # Hauptplot: y = 0.5 x, Limits 0..1 und 1:1 Aspect
    ax_main.plot(xs, 0.5 * xs, color="black", lw=1.5, label="y = 0.5 x")
    ax_main.set_xlim(0, 1)
    ax_main.set_ylim(0, 1)
    ax_main.set_xlabel("x")
    ax_main.set_ylabel("y")
    ax_main.set_title("Transformation y = 0.5 x (Bereiche 0..1)")
    ax_main.grid(True)
    ax_main.legend(loc="upper left", fontsize="small")
    ax_main.set_aspect("equal", adjustable="box")

    # Rand unten: p(x)
    line_px, = ax_px.plot(xs, p_x(xs, mu, sigma), color="tab:blue")
    ax_px.set_xlim(0, 1)
    ax_px.set_xlabel("x")
    ax_px.set_ylabel("p(x)")
    ax_px.grid(True)

    # Rand links: p(y) vertikal, horizontal gespiegelt (nach links)
    py_vals = -p_y(ys, mu, sigma, use_jacobian)  # gespiegelt, damit Kurve nach links zeigt
    line_py, = ax_py.plot(py_vals, ys, color="tab:orange")
    ax_py.set_ylim(0, 1)
    ax_py.set_xlabel("")  # Platz sparen
    ax_py.set_ylabel("y")
    ax_py.grid(True)

    # Feinheiten: gleiche padding, Achsenticks an passenden Seiten
    # Entferne x-ticks links oben (da negativ Werte)
    ax_py.tick_params(axis="x", which="both", labelbottom=False)
    # Entferne y-tick labels unten-right duplicate
    plt.setp(ax_px.get_xticklabels(), rotation=0)
    
    # ----- Checkbox für Jacobi-Faktor -----
    cb_ax = fig.add_axes([ax_check.get_position().x0 + 0.02,
                          ax_check.get_position().y0 + 0.02,
                          ax_check.get_position().width - 0.04,
                          ax_check.get_position().height - 0.04])
    check = CheckButtons(cb_ax, ["Jacobi an"], [use_jacobian])
    cb_ax.set_title("Optionen", fontsize=8)

    # ----- Update-Funktion für den Toggle -----
    def update(label):
        use_jac = check.get_status()[0]
        # Update p(y) (gespiegelt)
        line_py.set_xdata(-p_y(ys, mu, sigma, use_jac))
        fig.canvas.draw_idle()

    check.on_clicked(update)

    # Layout-Anpassungen
    ax_py.set_xticks([])

    # set z-order so margins don't overlap main plot visual
    ax_py.set_zorder(1)
    ax_main.set_zorder(2)
    ax_px.set_zorder(2)

    plt.show()

# ----- GUI mit Schiebereglern für mu und sigma -----
mu_slider = FloatSlider(value=0.75, min=0, max=1, step=0.01, description="Mittelwert (mu)")
sigma_slider = FloatSlider(value=0.12, min=0.01, max=0.5, step=0.01, description="Varianz (sigma)")
jacobi_toggle = ToggleButton(value=True, description="Jacobi Faktor", icon="check", tooltip="Aktivieren des Jacobi-Faktors")

# Interaktive Funktion
interactive_plot = interactive(update_plot, mu=mu_slider, sigma=sigma_slider, use_jacobian=jacobi_toggle)

# Anzeige der interaktiven Widgets
interactive_plot

