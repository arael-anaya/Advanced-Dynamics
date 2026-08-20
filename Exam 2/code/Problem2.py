import os
import matplotlib
matplotlib.use('Agg')
import numpy as np
from scipy.integrate import solve_ivp
from scipy.signal import find_peaks
import matplotlib.pyplot as plt

FIG_DIR = os.path.join(os.path.dirname(__file__), '..', 'latex', 'Figures', 'Graphs', 'Problem2')
os.makedirs(FIG_DIR, exist_ok=True)

g = 9.81


def build_system(state, r, M):
    """Assembles the 7x7 dynamics block A_7 (xdd, ydd, psdd, thdd, phdd, lam1, lam2 = np.linalg.solve(A_7, b_7)).
    This is the only nontrivial block of the full 12x12 system in part (e); the other 5 rows
    (qdot -> qdot) are just an identity pass-through and aren't worth assembling as a linear solve."""

    th, ps, ph, thd, psd, phd = state
    c, s = np.cos, np.sin
    ct, st = c(th), s(th)
    cp, sp = c(ps), s(ps)

    A = np.zeros((7, 7))
    b = np.zeros(7)

    # x: M*xdd = lam1
    A[0, 0] = M
    A[0, 5] = -1.0

    # y: M*ydd = lam2
    A[1, 1] = M
    A[1, 6] = -1.0

    # psi
    A[2, 2] = M * r**2 * (ct**2 + 1) / 4
    A[2, 4] = M * r**2 * ct / 2
    A[2, 5] = -r * ct * cp
    A[2, 6] = -r * ct * sp
    b[2] = M * r**2 * (st * thd / 2) * (phd + psd * ct)

    # theta
    A[3, 3] = M * r**2 * (ct**2 + 0.25)
    A[3, 5] = r * st * sp
    A[3, 6] = -r * st * cp
    b[3] = (-M * r**2 * st * ct * (psd**2 / 4 - thd**2)
            - 0.5 * M * r**2 * st * phd * psd
            - M * g * r * ct)

    # phi
    A[4, 2] = M * r**2 * ct / 2
    A[4, 4] = M * r**2 / 2
    A[4, 5] = -r * cp
    A[4, 6] = -r * sp
    b[4] = M * r**2 / 2 * psd * thd * st

    # d/dt(phi1) = 0
    A[5, 0] = 1.0
    A[5, 2] = r * cp * ct
    A[5, 3] = -r * sp * st
    A[5, 4] = r * cp
    b[5] = (r * sp * ct * (psd**2 + thd**2) + r * sp * phd * psd
            + 2 * r * st * cp * psd * thd)

    # d/dt(phi2) = 0
    A[6, 1] = 1.0
    A[6, 2] = r * sp * ct
    A[6, 3] = r * st * cp
    A[6, 4] = r * sp
    b[6] = (2 * r * sp * st * psd * thd - r * cp * ct * (psd**2 + thd**2)
            - r * cp * phd * psd)

    return A, b


def deriv(t, state, r, M):
    x, y, th, ps, ph, xd, yd, thd, psd, phd = state
    A, b = build_system((th, ps, ph, thd, psd, phd), r, M)
    xdd, ydd, psdd, thdd, phdd, lam1, lam2 = np.linalg.solve(A, b)
    return [xd, yd, thd, psd, phd, xdd, ydd, thdd, psdd, phdd]


r, M = 1.0, 1.0

theta0, psi0, phi0 = 1.2, 0.0, 0.0
thetadot0, psidot0 = 0.0, 1.5
x0, y0 = 0.0, 0.0


def constrained_ic(phidot0):
    xdot0 = -r * (phidot0 + psidot0 * np.cos(theta0)) * np.cos(psi0) + r * thetadot0 * np.sin(theta0) * np.sin(psi0)
    ydot0 = -r * (phidot0 + psidot0 * np.cos(theta0)) * np.sin(psi0) - r * thetadot0 * np.sin(theta0) * np.cos(psi0)
    return [x0, y0, theta0, psi0, phi0, xdot0, ydot0, thetadot0, psidot0, phidot0]


t_end = 15.0
t_eval = np.linspace(0, t_end, 6000)

# --- Zero-spin comparison: same theta0, psidot0, but phidot0 = 0 ---
state0_zerospin = constrained_ic(0.0)
sol0 = solve_ivp(deriv, [0, t_end], state0_zerospin, args=(r, M), t_eval=t_eval,
                  method='DOP853', rtol=1e-11, atol=1e-13)
theta_zerospin = sol0.y[2]
troughs, _ = find_peaks(-theta_zerospin)
period_zerospin = np.mean(np.diff(sol0.t[troughs]))
print(f"phidot0=0: theta oscillates periodically between {theta_zerospin.min():.3f} and {theta_zerospin.max():.3f} rad, "
      f"period approx {period_zerospin:.2f} s "
      f"(z=r*sin(theta) between {r*np.sin(theta_zerospin.min()):.3f} and {r*np.sin(theta_zerospin.max()):.3f})")

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(sol0.t, theta_zerospin)
ax.set_xlabel('t (s)'); ax.set_ylabel(r'$\theta$ (rad)')
ax.set_title(r'Zero-spin comparison ($\dot\phi_0=0$): $\theta$ vs t')
fig.tight_layout(); fig.savefig(os.path.join(FIG_DIR, 'partF_zerospin_theta_vs_t.png'), dpi=140)

# --- Main run: phidot0 = 6, giving a meaningful rolling/nutating solution ---
phidot0 = 6.0
state0 = constrained_ic(phidot0)

sol = solve_ivp(deriv, [0, t_end], state0, args=(r, M), t_eval=t_eval,
                method='DOP853', rtol=1e-11, atol=1e-13, dense_output=True)
x, y, theta, psi, phi, xd, yd, thd, psd, phd = sol.y


It, Ia = 0.25 * M * r**2, 0.5 * M * r**2
ct, st = np.cos(theta), np.sin(theta)
cph, sph = np.cos(phi), np.sin(phi)
cps, sps = np.cos(psi), np.sin(psi)
w1 = thd * cph + psd * st * sph
w2 = -thd * sph + psd * st * cph
w3 = phd + psd * ct
T = 0.5 * M * (xd**2 + yd**2 + (r * ct * thd)**2) + 0.5 * It * (w1**2 + w2**2) + 0.5 * Ia * w3**2
V = M * g * r * st
E = T + V
print(f"max relative drift in energy E: {np.max(np.abs(E - E[0])) / abs(E[0]):.3e}")

phi1 = xd + r * (phd + psd * ct) * cps - r * thd * st * sps
phi2 = yd + r * (phd + psd * ct) * sps + r * thd * st * cps
print(f"max |phi1|, |phi2| (rolling constraint residual): {np.max(np.abs(phi1)):.3e}, {np.max(np.abs(phi2)):.3e}")

v0 = r * abs(phidot0 + psidot0 * np.cos(theta0))
speed = np.sqrt(xd**2 + yd**2)
print(f"v at t=0: {v0:.3f} m/s;  mean speed over run: {speed.mean():.3f} m/s")

# arc length / total heading change: both time-averaged, so the instantaneous v0
# is never mixed with a time-averaged rate
trapz = np.trapezoid if hasattr(np, 'trapezoid') else np.trapz
arc_length = trapz(speed, sol.t)
delta_psi = psi[-1] - psi[0]
R_arc = arc_length / abs(delta_psi)
print(f"arc length: {arc_length:.3f} m;  total heading change |Delta psi|: {abs(delta_psi):.3f} rad;  "
      f"R = arc_length/|Delta psi| = {R_arc:.3f} m")
print(f"mean psidot over run: {psd.mean():.3f} rad/s;  mean(v)/mean(psidot) = {speed.mean()/psd.mean():.3f} m")

# algebraic circle fit (Kasa) to the x,y trajectory, for the actual turning radius
Amat = np.column_stack([x, y, np.ones_like(x)])
bvec = x**2 + y**2
sol_fit, *_ = np.linalg.lstsq(Amat, bvec, rcond=None)
xc, yc = sol_fit[0] / 2, sol_fit[1] / 2
R_fit = np.sqrt(sol_fit[2] + xc**2 + yc**2)
print(f"circle fit: center=({xc:.3f}, {yc:.3f}), radius={R_fit:.3f} m")

fig, ax = plt.subplots(figsize=(6, 6))
ax.plot(x, y)
ax.set_xlabel('x (m)'); ax.set_ylabel('y (m)'); ax.set_aspect('equal')
ax.set_title('Part (f): center-of-mass trajectory')
fig.tight_layout(); fig.savefig(os.path.join(FIG_DIR, 'partF_xy_trajectory.png'), dpi=140)

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(sol.t, theta)
ax.set_xlabel('t (s)'); ax.set_ylabel(r'$\theta$ (rad)')
ax.set_title(r'Part (f): $\theta$ vs t')
fig.tight_layout(); fig.savefig(os.path.join(FIG_DIR, 'partF_theta_vs_t.png'), dpi=140)
print("done")
