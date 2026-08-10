import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

FIG_DIR = os.path.join(os.path.dirname(__file__), "..", "latex", "Figures", "Phase Portraits")
os.makedirs(FIG_DIR, exist_ok=True)

h = 0.01

STYLE = {
    "saddle": dict(marker="s", mfc="white", mec="k", label="saddle (unstable)"),
    "stable node": dict(marker="o", mfc="0.5", mec="k", label="stable node"),
    "stable spiral": dict(marker="o", mfc="k", mec="k", label="stable spiral"),
}


def savefig(fig, name):
    fig.savefig(os.path.join(FIG_DIR, name), dpi=200, bbox_inches="tight")


def rhs(t, state, a):
    x, v = state
    return [v, -v + a * x - x**3 + h]


def equilibria(a, tol=1e-8):
    roots = np.roots([1.0, 0.0, -a, -h])
    real = np.sort(roots[np.abs(roots.imag) < tol].real)
    return real


def classify(xstar, a):
    Delta = 3 * xstar**2 - a
    if Delta < 0:
        return "saddle"
    elif Delta <= 0.25:
        # Delta == 0.25 exactly is the degenerate-node boundary, a
        # measure-zero set of (a, x*) that never actually occurs on the
        # sampled grid; bucketing it with "stable node" is harmless.
        return "stable node"
    else:
        return "stable spiral"


def mark_equilibrium(ax, xstar, kind):
    st = STYLE[kind]
    ax.plot(xstar, 0, st["marker"], mfc=st["mfc"], mec=st["mec"], ms=8, mew=1.3, zorder=5)


a = -1.0
X, V = np.meshgrid(np.linspace(-2, 2, 400), np.linspace(-2, 2, 400))
dX = V
dV = -V + a * X - X**3 + h

fig, ax = plt.subplots(figsize=(6, 5))
ax.streamplot(X, V, dX, dV, density=1.4, color="gray", linewidth=0.7)

roots = equilibria(a)
for xstar in roots:
    kind = classify(xstar, a)
    mark_equilibrium(ax, xstar, kind)
    ax.annotate(f"{kind}\n$x^*={xstar:.3f}$", (xstar, 0), xytext=(xstar + 0.15, 1.6), fontsize=9)

for x0 in ([1.8, 0], [-1.8, 1.0], [0, -1.8], [1.2, 1.5]):
    sol = solve_ivp(rhs, [0, 30], x0, args=(a,), max_step=0.02)
    ax.plot(sol.y[0], sol.y[1], color="C0", lw=1.2)

ax.axhline(0, lw=0.5, c="k")
ax.axvline(0, lw=0.5, c="k")
ax.set_xlabel("x")
ax.set_ylabel("v")
ax.set_title(f"Problem 4(b): $h={h}$, $a={a:g}$")
ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)
savefig(fig, "Problem4b-am1.png")
plt.close(fig)


a = 1.0
X, V = np.meshgrid(np.linspace(-2.5, 2.5, 400), np.linspace(-2.5, 2.5, 400))
dX = V
dV = -V + a * X - X**3 + h

fig, ax = plt.subplots(figsize=(7, 5.5))
ax.streamplot(X, V, dX, dV, density=1.4, color="gray", linewidth=0.7)

roots = equilibria(a)
kinds = [classify(xstar, a) for xstar in roots]
for xstar, kind in zip(roots, kinds):
    mark_equilibrium(ax, xstar, kind)
    ax.annotate(f"$x^*={xstar:.3f}$\n({kind})", (xstar, 0), xytext=(xstar - 0.15, 1.9 if xstar < 0 else -2.2),
                fontsize=8, ha="center")

saddle_idx = kinds.index("saddle")
xs = roots[saddle_idx]
Delta = 3 * xs**2 - a
lam = np.roots([1, 1, Delta])
lam_unstable = lam[lam.real > 0][0].real
lam_stable = lam[lam.real < 0][0].real

eps = 1e-3
for sign in (+1, -1):
    v0 = sign * eps * np.array([1, lam_unstable])
    sol = solve_ivp(rhs, [0, 30], [xs + v0[0], v0[1]], args=(a,), max_step=0.02)
    ax.plot(sol.y[0], sol.y[1], "--", color="C3", lw=1.5,
             label="unstable manifold" if sign == 1 else None)

    v0 = sign * eps * np.array([1, lam_stable])
    sol = solve_ivp(rhs, [0, -30], [xs + v0[0], v0[1]], args=(a,), max_step=0.02)
    ax.plot(sol.y[0], sol.y[1], "--", color="C0", lw=1.5,
             label="stable manifold" if sign == 1 else None)

for x0 in ([2.3, 1.0], [-2.3, -1.0], [0, 2.2], [0, -2.2]):
    sol = solve_ivp(rhs, [0, 30], x0, args=(a,), max_step=0.02)
    ax.plot(sol.y[0], sol.y[1], color="0.3", lw=1.0, alpha=0.7)

ax.axhline(0, lw=0.5, c="k")
ax.axvline(0, lw=0.5, c="k")
ax.set_xlabel("x")
ax.set_ylabel("v")
ax.set_title(f"Problem 4(b): $h={h}$, $a={a:g}$")
ax.set_xlim(-2.5, 2.5)
ax.set_ylim(-2.5, 2.5)
ax.legend(loc="upper left", fontsize=8)
savefig(fig, "Problem4b-ap1.png")
plt.close(fig)


a_vals = np.linspace(-1, 1, 400)
points = {"saddle": ([], []), "stable node": ([], []), "stable spiral": ([], [])}
for a_i in a_vals:
    for xstar in equilibria(a_i):
        kind = classify(xstar, a_i)
        points[kind][0].append(a_i)
        points[kind][1].append(xstar)

scatter_colors = {"saddle": "C3", "stable node": "C1", "stable spiral": "C0"}
fig, ax = plt.subplots(figsize=(6.5, 5))
for kind, (a_pts, x_pts) in points.items():
    ax.scatter(a_pts, x_pts, s=8, color=scatter_colors[kind], label=STYLE[kind]["label"])

ax.axhline(0, lw=0.5, c="k")
ax.axvline(0, lw=0.5, c="k")

# Saddle-node (fold) bifurcation point: tangency of x*^3 - a x* - h = 0 with
# Delta = 3x*^2 - a = 0 simultaneously. Eliminating x* = -sqrt(a/3) gives
# h = (2a/3) sqrt(a/3); solve numerically for a_c, then back out x*_c.
from scipy.optimize import brentq
fold_eq = lambda a_: (2 * a_ / 3) * np.sqrt(a_ / 3) - h
a_c = brentq(fold_eq, 1e-6, 1.0)
x_c = -np.sqrt(a_c / 3)
ax.plot(a_c, x_c, "*", color="k", ms=14, zorder=6)
ax.annotate(rf"fold: $a_c\approx{a_c:.4f}$" + "\n" + rf"$x^*_c\approx{x_c:.3f}$",
            (a_c, x_c), xytext=(a_c + 0.08, x_c - 0.35), fontsize=8,
            arrowprops=dict(arrowstyle="->", lw=0.8))

ax.set_xlabel("a")
ax.set_ylabel(r"$x^*$")
ax.set_title(f"Problem 4(e): equilibria vs. $a$, $h={h}$")
ax.legend(loc="upper left", fontsize=8, markerscale=2)
savefig(fig, "Problem4e-roots.png")
plt.close(fig)

print(f"Fold bifurcation: a_c={a_c:.4f}, x*_c={x_c:.4f}")
print(f"Saved Problem4b-am1/4b-ap1/4e-roots.png to: {os.path.abspath(FIG_DIR)}")
