import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

theta = np.deg2rad(60)
omega = 2
g = 9.81
r_dot_init = 0
r_span = (0.5 , 5)
t_span =(0,1.5)
m = 1

def system(t, state):
    r, v = state
    drdt = v
    dvdt =  r * omega**2 *(np.sin(theta))**2 - g * np.cos(theta)
    return [drdt , dvdt]

t_eval = np.linspace(t_span[0], t_span[-1], 100)
r_eval = np.linspace(r_span[0] , r_span[-1],10)

fig, (ax_r, ax_fn) = plt.subplots(1, 2, figsize=(13, 5.5))
colors = plt.cm.viridis(np.linspace(0, 1, len(r_eval)))

for r, color in zip(r_eval, colors):
    solution = solve_ivp(system, t_span, (r, r_dot_init), t_eval=t_eval)
    t_points = solution.t
    r_values = solution.y[0]
    v_values = solution.y[1]

    F_N_theta = -m * np.sin(theta) * (omega**2 * r_values * np.cos(theta) + g)
    F_N_phi = 2 * m * omega * np.sin(theta) * v_values
    F_N = np.sqrt(F_N_theta**2 + F_N_phi**2)

    label = f'$r_0$ = {r:.2f} m'
    ax_r.plot(t_points, r_values, label=label, color=color)
    ax_fn.plot(t_points, F_N, label=label, color=color)

ax_r.set_xlabel('Time (s)')
ax_r.set_ylabel('r (m)')
ax_r.set_title(r'Radial position $r(t)$')
ax_r.grid(True, alpha=0.3)

ax_fn.set_xlabel('Time (s)')
ax_fn.set_ylabel(r'$F_N$ (N)')
ax_fn.set_title(r'Normal force magnitude $F_N(t)$')
ax_fn.grid(True, alpha=0.3)

handles, labels = ax_r.get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=5, bbox_to_anchor=(0.5, -0.05))
fig.suptitle(r'Bead in rotating tube: $\theta = 60°$, $\Omega = 2$ rad/s', fontsize=13)
fig.tight_layout(rect=[0, 0.05, 1, 1])
plt.show()
