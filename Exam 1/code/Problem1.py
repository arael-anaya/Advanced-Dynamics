import os
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

FIG_DIR = os.path.join(os.path.dirname(__file__), '..', 'latex', 'Figures', 'Graphs', 'Problem1')
os.makedirs(FIG_DIR, exist_ok=True)

g = 9.81


def hamilton_deriv(t, y, m, M, L, g):
    theta, phi, p_theta, p_phi = y
    c, s = np.cos(theta), np.sin(theta)
    theta_dot = p_theta / (2*L**2*(m + 2*M*c**2))
    phi_dot = p_phi / (2*m*L**2*c**2)
    p_theta_dot = (-M*p_theta**2*s*c/(L**2*(m + 2*M*c**2)**2)
                    - p_phi**2*s/(2*m*L**2*c**3)
                    + 2*(m+M)*g*L*c)
    return [theta_dot, phi_dot, p_theta_dot, 0.0]

m, M, L = 1.0, 2.0, 1.0
theta0, phi0, ptheta0, pphi0 = 0.3, 0.0, 0.0, 3.0
t_end_g = 10.0
t_eval_g = np.linspace(0, t_end_g, 3000)

sol = solve_ivp(hamilton_deriv, [0, t_end_g], [theta0, phi0, ptheta0, pphi0],
                args=(m, M, L, g), t_eval=t_eval_g, method='DOP853', rtol=1e-10, atol=1e-12)
theta, phi, p_theta, p_phi = sol.y
phi_dot = p_phi / (2*m*L**2*np.cos(theta)**2)

H = (p_theta**2/(4*L**2*(m + 2*M*np.cos(theta)**2)) + p_phi**2/(4*m*L**2*np.cos(theta)**2)
     - 2*(m+M)*g*L*np.sin(theta))
print(f"max relative drift in H: {np.max(np.abs(H - H[0]))/abs(H[0]):.3e}")

fig, ax = plt.subplots(4, 1, sharex=True, figsize=(8, 9))
ax[0].plot(sol.t, theta); ax[0].set_ylabel(r'$\theta$ (rad)')
ax[1].plot(sol.t, phi); ax[1].set_ylabel(r'$\phi$ (rad)')
ax[2].plot(sol.t, p_theta); ax[2].set_ylabel(r'$p_\theta$')
ax[3].plot(sol.t, p_phi); ax[3].set_ylabel(r'$p_\phi$'); ax[3].set_xlabel('t (s)')
ax[0].set_title('Part (g): generalized coordinates and momenta vs time')
fig.tight_layout(); fig.savefig(os.path.join(FIG_DIR, 'partG_qp_vs_t.png'), dpi=140)

fig, ax = plt.subplots(figsize=(6, 5))
ax.plot(theta, phi_dot)
ax.set_xlabel(r'$\theta$ (rad)'); ax.set_ylabel(r'$\dot\phi$ (rad/s)')
ax.set_title(r'Part (g): $\theta$ vs $\dot\phi$')
fig.tight_layout(); fig.savefig(os.path.join(FIG_DIR, 'partG_theta_vs_phidot.png'), dpi=140)

plt.show()
