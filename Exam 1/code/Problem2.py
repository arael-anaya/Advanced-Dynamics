import os
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

FIG_DIR = os.path.join(os.path.dirname(__file__), '..', 'latex', 'Figures', 'Graphs', 'Problem2')
os.makedirs(FIG_DIR, exist_ok=True)


def deriv(t, state, M, I, d, l, omega0):
    v, theta, x, y = state
    psi = (np.pi/6)*np.cos(omega0*t)
    psi_dot = -(np.pi/6)*omega0*np.sin(omega0*t)
    psi_ddot = -(np.pi/6)*omega0**2*np.cos(omega0*t)
    c, s = np.cos(psi), np.sin(psi)
    denom = d*c - l

    theta_dot = (l*psi_dot + v*s)/denom
    theta_ddot = ((denom*(v*psi_dot*c + l*psi_ddot) + d*psi_dot*s*(v*s + l*psi_dot))
                    / (denom**2 + (I/M)*s**2))
    v_dot = -I*theta_ddot*s/(M*denom)

    return [v_dot, theta_dot, v*np.cos(theta), v*np.sin(theta)]

M, I, d, l, omega0 = 1.0, 0.5, 1.0, 0.5, 1.0
t_end = 20.0
t_eval = np.linspace(0, t_end, 4000)

sol = solve_ivp(deriv, [0, t_end], [0.0, 0.0, 0.0, 0.0], args=(M, I, d, l, omega0),
                t_eval=t_eval, method='DOP853', rtol=1e-10, atol=1e-12, dense_output=True)
v, theta, x, y = sol.y

t_f = np.linspace(0, t_end, 200001)
v_f, th_f, x_f, y_f = sol.sol(t_f)
psi_f, psid_f = (np.pi/6)*np.cos(omega0*t_f), -(np.pi/6)*omega0*np.sin(omega0*t_f)
xd_f, yd_f, thd_f = np.gradient(x_f, t_f), np.gradient(y_f, t_f), np.gradient(th_f, t_f)
phi1 = -xd_f*np.sin(th_f) + yd_f*np.cos(th_f)
phi2 = -np.sin(psi_f)*(xd_f*np.cos(th_f)+yd_f*np.sin(th_f)) + (d*np.cos(psi_f)-l)*thd_f - l*psid_f
print(f"max |phi1|, |phi2|: {np.max(np.abs(phi1)):.3e}, {np.max(np.abs(phi2)):.3e}")

fig, ax = plt.subplots(figsize=(6, 6))
ax.plot(x, y)
ax.set_xlabel('x (m)'); ax.set_ylabel('y (m)'); ax.set_aspect('equal')
ax.set_title('Part (d): center-of-mass trajectory')
fig.tight_layout(); fig.savefig(os.path.join(FIG_DIR, 'partD_xy_trajectory.png'), dpi=140)

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(sol.t, theta)
ax.set_xlabel('t (s)'); ax.set_ylabel(r'$\theta$ (rad)')
ax.set_title(r'Part (d): $\theta$ vs t')
fig.tight_layout(); fig.savefig(os.path.join(FIG_DIR, 'partD_theta_vs_t.png'), dpi=140)

plt.show()
