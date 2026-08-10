import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

FIG_DIR = os.path.join(os.path.dirname(__file__), "..", "latex", "Figures", "Phase Portraits")
os.makedirs(FIG_DIR, exist_ok=True)


def savefig(fig, name):
    fig.savefig(os.path.join(FIG_DIR, name), dpi=200, bbox_inches="tight")


M, k = 1.0, 1.0
omega = np.sqrt(k / M)

X, V = np.meshgrid(np.linspace(-2, 2, 400), np.linspace(-2, 2, 400))
dX = V
dV = -(k / M) * X

fig, ax = plt.subplots(figsize=(6, 5))
ax.streamplot(X, V, dX, dV, density=1.4, color="gray", linewidth=0.7)

theta = np.linspace(0, 2 * np.pi, 200)
for r in (0.5, 1.0, 1.5, 2.0):
    ax.plot(r * np.cos(theta), omega * r * np.sin(theta), color="C0", lw=1.3)

ax.plot(0, 0, "ko", ms=6)
ax.annotate("center", (0, 0), xytext=(0.15, 1.85), fontsize=10)
ax.axhline(0, lw=0.5, c="k")
ax.axvline(0, lw=0.5, c="k")
ax.set_xlabel("x")
ax.set_ylabel("v")
ax.set_title("Problem 1(a): frictionless sliding")
ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)
savefig(fig, "Problem1a.png")
plt.close(fig)


b_cr = 2 * np.sqrt(k * M)


def damped_rhs(t, state, b):
    x, v = state
    return [v, -(k / M) * x - (b / M) * v]


fig, (ax_spiral, ax_crit, ax_node) = plt.subplots(1, 3, figsize=(16, 5))

b_spiral = 0.5 * b_cr
X, V = np.meshgrid(np.linspace(-2, 2, 400), np.linspace(-2, 2, 400))
dX = V
dV = -(k / M) * X - (b_spiral / M) * V

ax_spiral.streamplot(X, V, dX, dV, density=1.4, color="gray", linewidth=0.7)

for x0 in ([1.8, 0], [-1.5, 1.0], [0, -1.8]):
    sol = solve_ivp(damped_rhs, [0, 30], x0, args=(b_spiral,), max_step=0.02)
    ax_spiral.plot(sol.y[0], sol.y[1], color="C0", lw=1.3)

ax_spiral.plot(0, 0, "ko", ms=6)
ax_spiral.annotate("stable spiral", (0, 0), xytext=(0.15, 1.85), fontsize=10)
ax_spiral.axhline(0, lw=0.5, c="k")
ax_spiral.axvline(0, lw=0.5, c="k")
ax_spiral.set_xlabel("x")
ax_spiral.set_ylabel("v")
ax_spiral.set_title(rf"$b={b_spiral:.2g} < b_{{cr}}={b_cr:.2g}$ (spiral)")
ax_spiral.set_xlim(-2, 2)
ax_spiral.set_ylim(-2, 2)

lam_crit = -np.sqrt(k / M)

X, V = np.meshgrid(np.linspace(-2, 2, 400), np.linspace(-2, 2, 400))
dX = V
dV = -(k / M) * X - (b_cr / M) * V

ax_crit.streamplot(X, V, dX, dV, density=1.4, color="gray", linewidth=0.7)

xs = np.linspace(-2, 2, 50)
ax_crit.plot(xs, lam_crit * xs, "--", color="C1", lw=1.5,
             label=f"coalesced eigendirection ($\\lambda={lam_crit:.2f}$)")

for x0 in ([1.8, -0.5], [-1.8, 0.5], [1.0, 1.8], [-1.0, -1.8]):
    sol = solve_ivp(damped_rhs, [0, 30], x0, args=(b_cr,), max_step=0.02)
    ax_crit.plot(sol.y[0], sol.y[1], color="C0", lw=1.2)

ax_crit.plot(0, 0, "ko", ms=6)
ax_crit.annotate("degenerate node", (0, 0), xytext=(0.15, 1.85), fontsize=10)
ax_crit.axhline(0, lw=0.5, c="k")
ax_crit.axvline(0, lw=0.5, c="k")
ax_crit.set_xlabel("x")
ax_crit.set_ylabel("v")
ax_crit.set_title(rf"$b=b_{{cr}}={b_cr:.2g}$ (critically damped)")
ax_crit.set_xlim(-2, 2)
ax_crit.set_ylim(-2, 2)
ax_crit.legend(loc="lower right", fontsize=8)

b_node = 1.5 * b_cr
lam = np.roots([1, b_node / M, k / M])
lam_slow, lam_fast = sorted(lam, key=abs)

X, V = np.meshgrid(np.linspace(-2, 2, 400), np.linspace(-2, 2, 400))
dX = V
dV = -(k / M) * X - (b_node / M) * V

ax_node.streamplot(X, V, dX, dV, density=1.4, color="gray", linewidth=0.7)

xs = np.linspace(-2, 2, 50)
ax_node.plot(xs, lam_slow * xs, "--", color="C1", lw=1.5, label=f"slow eigendirection ($\\lambda={lam_slow:.2f}$)")
ax_node.plot(xs, lam_fast * xs, "--", color="C3", lw=1.5, label=f"fast eigendirection ($\\lambda={lam_fast:.2f}$)")

for x0 in ([1.8, -0.5], [-1.8, 0.5], [1.0, 1.8], [-1.0, -1.8]):
    sol = solve_ivp(damped_rhs, [0, 30], x0, args=(b_node,), max_step=0.02)
    ax_node.plot(sol.y[0], sol.y[1], color="C0", lw=1.2)

ax_node.plot(0, 0, "ko", ms=6)
ax_node.annotate("stable node", (0, 0), xytext=(0.15, 1.85), fontsize=10)
ax_node.axhline(0, lw=0.5, c="k")
ax_node.axvline(0, lw=0.5, c="k")
ax_node.set_xlabel("x")
ax_node.set_ylabel("v")
ax_node.set_title(rf"$b={b_node:.2g} > b_{{cr}}={b_cr:.2g}$ (node)")
ax_node.set_xlim(-2, 2)
ax_node.set_ylim(-2, 2)
ax_node.legend(loc="lower right", fontsize=8)

fig.suptitle("Problem 1(b): linear damper")
fig.tight_layout()
savefig(fig, "Problem1b.png")
plt.close(fig)


muN = 1.0

def coulomb_rhs(t, state):
    x, v = state
    if abs(v) < 1e-3 and abs(k * x) <= muN:
        return [0.0, 0.0]
    fric = muN if v > 0 else (-muN if v < 0 else 0.0)
    return [v, -(k / M) * x - fric / M]


X, V = np.meshgrid(np.linspace(-6, 6, 500), np.linspace(-5, 5, 500))
dX = V
dV = -(k / M) * X - (muN / M) * np.sign(V)

fig, ax = plt.subplots(figsize=(7, 5.5))
ax.streamplot(X, V, dX, dV, density=1.4, color="gray", linewidth=0.7)

ax.axvspan(-muN / k, muN / k, color="0.85", zorder=0)
ax.annotate("dead band", (0, 4.4), ha="center", fontsize=9)

for x0 in ([5.5, 0.0], [-5.5, 0.0]):
    sol = solve_ivp(coulomb_rhs, [0, 60], x0, max_step=0.005, dense_output=False)
    ax.plot(sol.y[0], sol.y[1], color="C0", lw=1.3)

ax.plot([-muN / k, muN / k], [0, 0], "ko", ms=5)
ax.annotate(r"centers at $\pm\mu N/k$", (muN / k, 0), xytext=(2.3, -3.0), fontsize=9,
            arrowprops=dict(arrowstyle="->", lw=0.8))
ax.axhline(0, lw=0.5, c="k")
ax.axvline(0, lw=0.5, c="k")
ax.set_xlabel("x")
ax.set_ylabel("v")
ax.set_title("Problem 1(c): Coulomb friction")
ax.set_xlim(-6, 6)
ax.set_ylim(-5, 5)
savefig(fig, "Problem1c.png")
plt.close(fig)

print(f"Saved Problem1a/1b/1c.png to: {os.path.abspath(FIG_DIR)}")
