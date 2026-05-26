import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.widgets import Slider

# ---------------- Dynamik ----------------
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
P = np.array([[sigma_state**2, 0],[0, sigma_state**2]])
Q = np.array([[0.01,0],[0,0.01]])
R = np.array([[0.05,0],[0,0.05]])
state = np.array([0.5,0.5])
traj = [state.copy()]
observations = []

# ---------------- Plot ----------------
fig, ax = plt.subplots(figsize=(8,8))
plt.subplots_adjust(bottom=0.3)  # Platz für Slider
ax.set_xlim(-3,3)
ax.set_ylim(-3,3)
ax.set_aspect('equal')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.grid(True)
ax.set_title('EKF: Prediction (orange) → Klick → Update')

# Vektorfeld
XX_v, YY_v = np.meshgrid(np.linspace(-3,3,20), np.linspace(-3,3,20))
UU, VV = np.zeros_like(XX_v), np.zeros_like(YY_v)
for i in range(XX_v.shape[0]):
    for j in range(XX_v.shape[1]):
        vec = f([XX_v[i,j], YY_v[i,j]])
        UU[i,j] = vec[0]-XX_v[i,j]
        VV[i,j] = vec[1]-YY_v[i,j]
ax.quiver(XX_v, YY_v, UU, VV, color='gray', alpha=0.6)

# ---------------- Plot Elemente ----------------
traj_line, = ax.plot([state[0]], [state[1]], 'k-o', label='Trajektorie')
obs_scatter = ax.scatter([],[],color='red',label='Beobachtungen')
ellipse_patch = covariance_ellipse(state, P, edgecolor='blue', facecolor='none', linewidth=2)
ax.add_patch(ellipse_patch)
pred_scatter = ax.scatter([],[], color='orange', marker='x', s=80, label='Prediction')

# Beobachtungsunsicherheit (rot gestrichelt)
R_ellipse_patch = covariance_ellipse(state, R, edgecolor='red', facecolor='none', linewidth=1.5, linestyle='--')
ax.add_patch(R_ellipse_patch)

ax.legend()

# ---------------- EKF ----------------
def ekf_predict(x, P):
    F = jacobian(x)
    x_pred = f(x)
    P_pred = F @ P @ F.T + Q
    return x_pred, P_pred

def ekf_update(x_pred, P_pred, z):
    global R
    H = np.eye(2)
    y = z - x_pred
    S = H @ P_pred @ H.T + R
    K = P_pred @ H.T @ np.linalg.inv(S)
    x_upd = x_pred + K @ y
    P_upd = (np.eye(2) - K @ H) @ P_pred
    return x_upd, P_upd

# ---------------- Plot Update ----------------
def update_plot(x_display, P_display, pred_point):
    traj_arr = np.array(traj)
    traj_line.set_data(traj_arr[:,0], traj_arr[:,1])
    obs_arr = np.array(observations)
    if len(observations)>0:
        obs_scatter.set_offsets(obs_arr)

    # Zustandsellipse
    ellipse_patch.set_center(x_display)
    vals,vecs = np.linalg.eigh(P_display)
    order = vals.argsort()[::-1]
    vals=vals[order]
    vecs=vecs[:,order]
    angle=np.degrees(np.arctan2(vecs[1,0],vecs[0,0]))
    width,height = 2*2*np.sqrt(vals)
    ellipse_patch.width = width
    ellipse_patch.height = height
    ellipse_patch.angle = angle

    # Prediction Marker
    pred_scatter.set_offsets(pred_point.reshape(1,2))

    # Beobachtungsunsicherheit (R) rund um Prediction
    R_ellipse_patch.set_center(pred_point)
    vals, vecs = np.linalg.eigh(R)
    order = vals.argsort()[::-1]
    vals = vals[order]
    vecs = vecs[:, order]
    angle = np.degrees(np.arctan2(vecs[1,0], vecs[0,0]))
    width,height = 2*2*np.sqrt(vals)
    R_ellipse_patch.width = width
    R_ellipse_patch.height = height
    R_ellipse_patch.angle = angle

    fig.canvas.draw_idle()

# ---------------- Initial Prediction ----------------
pred_state, pred_P = ekf_predict(state, P)
update_plot(state, P, pred_state)

# ---------------- Mouse Click ----------------
def onclick(event):
    global state, P, traj, observations, pred_state, pred_P
    if event.inaxes != ax:
        return
    z = np.array([event.xdata,event.ydata])
    observations.append(z)

    # EKF Update
    state, P = ekf_update(pred_state, pred_P, z)
    traj.append(state.copy())

    # Next Prediction
    pred_state, pred_P = ekf_predict(state, P)
    update_plot(state, P, pred_state)

cid = fig.canvas.mpl_connect('button_press_event', onclick)

# ---------------- Slider UI ----------------
axcolor = 'lightgoldenrodyellow'
ax_R00 = plt.axes([0.15, 0.2, 0.65, 0.03], facecolor=axcolor)
ax_R11 = plt.axes([0.15, 0.15, 0.65, 0.03], facecolor=axcolor)
ax_R01 = plt.axes([0.15, 0.1, 0.65, 0.03], facecolor=axcolor)

s_R00 = Slider(ax_R00, 'R[0,0]', 0.001, 1.0, valinit=R[0,0])
s_R11 = Slider(ax_R11, 'R[1,1]', 0.001, 1.0, valinit=R[1,1])
s_R01 = Slider(ax_R01, 'R[0,1]=R[1,0]', -0.5, 0.5, valinit=R[0,1])

def update_R(val):
    global R
    R[0,0] = s_R00.val
    R[1,1] = s_R11.val
    R[0,1] = R[1,0] = s_R01.val
    # Update R-Ellipse sofort um Orange-Marker
    R_ellipse_patch.set_center(pred_state)
    vals, vecs = np.linalg.eigh(R)
    order = vals.argsort()[::-1]
    vals = vals[order]
    vecs = vecs[:, order]
    angle = np.degrees(np.arctan2(vecs[1,0], vecs[0,0]))
    width,height = 2*2*np.sqrt(vals)
    R_ellipse_patch.width = width
    R_ellipse_patch.height = height
    R_ellipse_patch.angle = angle
    fig.canvas.draw_idle()

s_R00.on_changed(update_R)
s_R11.on_changed(update_R)
s_R01.on_changed(update_R)

plt.show()
