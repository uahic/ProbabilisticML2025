import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import CheckButtons

# --- Originalverteilung p(x): Normalverteilung mit Mittelwert 2, Var=1 ---
def p_x(x):
    return 1/np.sqrt(2*np.pi) * np.exp(-(x-2)**2/2)

# --- Transformation: y = 0.5*x  =>  x = 2*y ---
def p_y(y, use_jacobian):
    x = 2 * y
    base = p_x(x)
    if use_jacobian:
        return 2 * base      # |dx/dy| = 2
    else:
        return base          # Jacobi-Faktor ausgeschaltet
        

# --- Datenbereiche ---
xs = np.linspace(-2, 6, 300)
ys = np.linspace(-1, 3, 300)

# --- Matplotlib Setup ---
fig, ax = plt.subplots(figsize=(8, 5))
plt.subplots_adjust(left=0.25)

# Linien zeichnen
line_px, = ax.plot(xs, p_x(xs), label="p(x)")
line_py, = ax.plot(ys, p_y(ys, True), label="p(y)")

ax.legend()
ax.set_title("Transformation y = 0.5 * x")
ax.set_xlabel("Value")
ax.set_ylabel("Density")
ax.grid(True)

# --- Checkbox einfügen ---
check_ax = plt.axes([0.02, 0.4, 0.15, 0.15])
check = CheckButtons(check_ax, ["Jacobi an"], [True])

# --- Update-Funktion für Checkbox ---
def update(label):
    use = check.get_status()[0]
    line_py.set_ydata(p_y(ys, use))
    plt.draw()

check.on_clicked(update)

plt.show()

