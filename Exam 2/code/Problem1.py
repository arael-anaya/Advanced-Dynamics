import os
import matplotlib
matplotlib.use('Agg')
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

FIG_DIR = os.path.join(os.path.dirname(__file__), '..', 'latex', 'Figures', 'Graphs', 'Problem1')
os.makedirs(FIG_DIR, exist_ok=True)

g, l = 9.81, 1.0
Omega_crit = np.sqrt(48/79 * g/l)
print(f"Omega_crit = {Omega_crit:.4f} rad/s")


def theta_ddot(theta, Omega):
    return (79/81)*Omega**2*np.sin(theta)*np.cos(theta) - (16/27)*(g/l)*np.sin(theta)


def deriv(t, state, Omega):
    theta, thetadot = state
    return [thetadot, theta_ddot(theta, Omega)]


def phase_portrait(ax, Omega, title):
    t_span = [0, 20]
    t_eval = np.linspace(*t_span, 4000)

    # librating orbits released from rest at various theta0
    for theta0 in np.linspace(-2.6, 2.6, 11):
        if abs(theta0) < 1e-3:
            continue
        sol = solve_ivp(deriv, t_span, [theta0, 0.0], args=(Omega,),
                         t_eval=t_eval, method='DOP853', rtol=1e-10, atol=1e-12)
        ax.plot(sol.y[0], sol.y[1], color='C0', lw=0.8)

    # trajectories launched from theta=0 with a range of angular velocities,
    # tracing out the separatrix and the circulating orbits beyond it
    for thetadot0 in [-2.5, -1.5, -0.6, 0.6, 1.5, 2.5]:
        sol = solve_ivp(deriv, t_span, [0.0, thetadot0], args=(Omega,),
                         t_eval=t_eval, method='DOP853', rtol=1e-10, atol=1e-12)
        ax.plot(sol.y[0], sol.y[1], color='C1', lw=0.8)

    ax.plot(0, 0, 'ko', ms=4)
    ax.set_xlabel(r'$\theta$ (rad)')
    ax.set_ylabel(r'$\dot\theta$ (rad/s)')
    ax.set_title(title)
    ax.set_xlim(-2.8, 2.8)


fig, axes = plt.subplots(1, 2, figsize=(11, 5))
phase_portrait(axes[0], 0.7*Omega_crit, r'$\Omega=0.7\,\Omega_{crit}$: $\theta=0$ a center')
phase_portrait(axes[1], 1.3*Omega_crit, r'$\Omega=1.3\,\Omega_{crit}$: $\theta=0$ a saddle')
fig.suptitle('Part (e): phase portrait before and after the bifurcation')
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, 'partE_phase_portrait.png'), dpi=140)
print("done")
