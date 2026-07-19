import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

g = 9.81
m = 1
k = 10
r_spring_0 = 1
r_dot_0 = 0
omega_0 = .01



t_span =(0,10)
theta_span = (0 , np.pi)

r_span = (r_spring_0 + (m*g) / k , r_spring_0 - (m*g) / k)


def system(t, state):
    r, v , theta, omega = state
    drdt = v
    dthetadt = omega

    dvdt = r * omega**2 + g*np.cos(theta) - (k/m)  * (r - r_spring_0)
    domegadt = - ( (g * np.sin(theta) + 2 * v * omega ) / r)


    return [drdt , dvdt , dthetadt , domegadt]

t_eval = np.linspace(t_span[0], t_span[-1], 1000)

cases = [
    dict(label=r'Near stable eq. ($\theta_e=0$)', theta0=theta_span[0], r0=r_span[0], r_dot0=r_dot_0, omega0=omega_0),
    dict(label=r'Near unstable eq. ($\theta_e=\pi$)', theta0=theta_span[1], r0=r_span[1], r_dot0=r_dot_0, omega0=omega_0),
    dict(label='Large swing (away from eq.)', theta0=np.pi / 2, r0=r_spring_0, r_dot0=0.5, omega0=1.0),
]

colors = plt.cm.viridis(np.linspace(0, 1, len(cases)))

fig, axes_r = plt.subplots(1, len(cases), figsize=(6 * len(cases), 6))
fig_E, axes_E = plt.subplots(1, len(cases), figsize=(6 * len(cases), 5))

for case, color, ax_r, ax_E in zip(cases, colors, axes_r, axes_E):
    solution = solve_ivp(system, t_span, (case['r0'], case['r_dot0'], case['theta0'], case['omega0']), t_eval=t_eval)
    t_points = solution.t
    r_values = solution.y[0]
    v_values = solution.y[1]
    theta_values = solution.y[2]
    omega_values = solution.y[3]

    T = (1/2) * m * (v_values**2 + (r_values * omega_values)**2)
    V = (1/2) * k * (r_values - r_spring_0)**2 - m * g * r_values * np.cos(theta_values)
    E = T + V

    x = r_values * np.sin(theta_values)

    y = - r_values * np.cos(theta_values)

    ax_r.plot(x, y, color=color, label=case['label'])
    ax_r.set_xlabel('x (m)')
    ax_r.set_ylabel('y (m)')
    ax_r.set_title(case['label'], fontsize=10)
    ax_r.grid(True, alpha=0.3)
    ax_r.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), fontsize=8, frameon=False)

    ax_E.plot(t_points, T, color=color, linestyle='-', label='Kinetic energy, T')
    ax_E.plot(t_points, V, color=color, linestyle='--', label='Potential energy, V')
    ax_E.plot(t_points, E, color=color, linestyle=':', label='Total energy, E')
    ax_E.set_xlabel('t (s)')
    ax_E.set_ylabel('Energy (J)')
    ax_E.set_title(case['label'], fontsize=10)
    ax_E.grid(True, alpha=0.3)
    ax_E.legend(loc='best', fontsize=8)


fig.suptitle(r'Elastic pendulum: mass trajectory ($r_0$ (spring) $= %.2f$ m, $k = %.2f$ N/m)' % (r_spring_0, k), fontsize=13)
fig.tight_layout(rect=[0, 0.08, 1, 0.95])

fig_E.suptitle(r'Elastic pendulum: kinetic, potential, and total energy ($r_0$ (spring) $= %.2f$ m, $k = %.2f$ N/m)' % (r_spring_0, k), fontsize=13)
fig_E.tight_layout()

plt.show()
