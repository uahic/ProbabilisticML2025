import pandas as pd
import numpy as np

# Zeitachse
n_points = 50 
t = np.linspace(0, 2, n_points)
dt = t[1] - t[0]

# Kurvige Trajektorie
x = np.sin(t) + 0.2*np.sin(5*t)   # Kombination aus großer und kleiner Sinuswelle
y = np.cos(t) + 0.2*np.cos(3*t)   # Kombination aus großer und kleiner Cosinuswelle

# Ableitungen (numerische Approximation)
xdot = np.gradient(x, dt)
ydot = np.gradient(y, dt)

# Daten in DataFrame
df = pd.DataFrame({
    't': t,
    'X': x,
    'dX': xdot,
    'Y': y,
    'dY': ydot
})

# CSV speichern
df.to_csv('trajectory_data.csv', index=False)
print("CSV 'trajectory_data.csv' erstellt.")
