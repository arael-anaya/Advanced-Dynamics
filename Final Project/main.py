import numpy as np
from scipy.integrate import solve_ivp

# Paramerts
m = 80
r_o = 1
k = 20000
g = 9.81
theta_TD = np.radians(68)
y_o = 1.2
v_o = 3.0

params = {'m' : m , 'k' : k , 'g' : g, 'r_o' : r_o}


def touchdownState(y_o , v_o, theta_TD, r_o, g):
    r = r_o
    theta = np.arctan2(r_o * np.sin(theta_TD) , -r_o * np.cos(theta_TD))

    x_TD_dot = v_o
    y_TD_dot = - (2*g*(y_o - r_o * np.sin(theta_TD)))**.5


    r_dot = x_TD_dot * np.cos(theta) + y_TD_dot * np.sin(theta)

    theta_dot = (-x_TD_dot * np.sin(theta) + y_TD_dot * np.cos(theta)) / r_o



    return [r , theta, r_dot , theta_dot]

def liftoff(t, s, params):
    return params['k'] * (s[0] - params['r_o'])
liftoff.terminal = True
liftoff.direction = 1


def stance_RHS(t , s, params):
    r_dot = s[2]
    theta_dot = s[3]


    r_ddot = s[0] * theta_dot**2 - params['g'] * np.sin(s[1]) - (params['k'] / params['m'])* (s[0] - params['r_o'])
    theta_ddot = -(2/s[0]) * r_dot * theta_dot - (params['g'] / s[0]) * np.cos(s[1])

    return [r_dot , theta_dot, r_ddot, theta_ddot]

def energy(s, params):
    E = (1/2) * params['m'] * (s[2]**2 + s[0]**2 * s[3]**2) + params['m'] * params['g'] * s[0] * np.sin(s[1]) + (1/2) * params['k']* (s[0] - params['r_o'])**2

    return E

s0 = touchdownState(y_o , v_o, theta_TD, r_o, g)
t_span = (0.0,1.0)

sol = solve_ivp(stance_RHS, t_span, s0, args=(params,),
                events=liftoff, rtol=1e-10, atol=1e-12,
                dense_output=True, max_step=1e-3)

E = np.array([energy(sol.y[:, i], params) for i in range(sol.y.shape[1])])
drift = np.max(np.abs(E - E[0])) / np.abs(E[0])

print("status:", sol.status)
print("liftoff time:", sol.t_events[0])
print("drift:", drift)

import matplotlib.pyplot as plt

# ---------- diagnostics ----------
r_TD, th_TD, rdot_TD, thdot_TD = s0
s_LO = sol.y_events[0][0]
t_LO = sol.t_events[0][0]

print("--- touchdown state ---")
print(f"  r      = {r_TD:.6f} m")
print(f"  theta  = {np.degrees(th_TD):.4f} deg")
print(f"  r_dot  = {rdot_TD:.6f} m/s")
print(f"  th_dot = {thdot_TD:.6f} rad/s")

print("--- liftoff state ---")
print(f"  status         = {sol.status}")
print(f"  stance duration= {t_LO:.6f} s")
print(f"  r              = {s_LO[0]:.6f} m")
print(f"  theta          = {np.degrees(s_LO[1]):.4f} deg")
print(f"  r_dot          = {s_LO[2]:.6f} m/s")
print(f"  th_dot         = {s_LO[3]:.6f} rad/s")

print("--- validation ---")
print(f"  min leg length = {np.min(sol.y[0]):.6f} m")
print(f"  max compression= {params['r_o'] - np.min(sol.y[0]):.6f} m")
print(f"  E_0            = {E[0]:.6f} J")
print(f"  relative drift = {drift:.3e}")

# ---------- figures ----------
t = sol.t
r = sol.y[0]
th = sol.y[1]
x_com = r * np.cos(th)
y_com = r * np.sin(th)

fig, ax = plt.subplots(2, 2, figsize=(11, 8))

ax[0, 0].plot(t, r, color='C3')
ax[0, 0].axhline(params['r_o'], ls='--', lw=0.8, color='gray', label='$r_0$')
ax[0, 0].set_xlabel('t [s]')
ax[0, 0].set_ylabel('r [m]')
ax[0, 0].set_title('Leg length')
ax[0, 0].legend()

ax[0, 1].plot(t, np.degrees(th), color='C0')
ax[0, 1].axhline(90, ls='--', lw=0.8, color='gray', label='$90^\\circ$')
ax[0, 1].set_xlabel('t [s]')
ax[0, 1].set_ylabel(r'$\theta$ [deg]')
ax[0, 1].set_title('Leg angle')
ax[0, 1].legend()

ax[1, 0].plot(x_com, y_com, color='C3', lw=2)
ax[1, 0].plot(0, 0, 'kv', ms=10, label='foot (pinned)')
ax[1, 0].plot(x_com[0], y_com[0], 'o', color='C0', label='touchdown')
ax[1, 0].plot(x_com[-1], y_com[-1], 's', color='C2', label='liftoff')
ax[1, 0].axhline(0, color='k', lw=1)
ax[1, 0].set_xlabel('x [m]')
ax[1, 0].set_ylabel('y [m]')
ax[1, 0].set_title('CoM trajectory (foot frame)')
ax[1, 0].axis('equal')
ax[1, 0].legend()

ax[1, 1].plot(t, E - E[0], color='C4')
ax[1, 1].set_xlabel('t [s]')
ax[1, 1].set_ylabel('$E - E_0$ [J]')
ax[1, 1].set_title(f'Energy drift (rel. {drift:.2e})')
ax[1, 1].ticklabel_format(axis='y', style='sci', scilimits=(0, 0))

for a in ax.flat:
    a.grid(alpha=0.3)
fig.tight_layout()
fig.savefig('stance_validation.png', dpi=200)
plt.show()