import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

FIG_DIR = os.path.join(os.path.dirname(__file__), "..", "latex", "Figures", "Phase Portraits")
os.makedirs(FIG_DIR, exist_ok=True)


def savefig(fig, name):
    fig.savefig(os.path.join(FIG_DIR, name), dpi=200, bbox_inches="tight")


def rhs(t, state, mu):
    x, v = state
    return [v, -x + mu * (1 - x**2) * v]


mu = -1.0
X, V = np.meshgrid(np.linspace(-3, 3, 400), np.linspace(-3, 3, 400))
dX = V
dV = -X + mu * (1 - X**2) * V

fig, ax = plt.subplots(figsize=(6, 5))
ax.streamplot(X, V, dX, dV, density=1.4, color="gray", linewidth=0.7)

# Seeds strictly inside the (unstable) limit cycle so they genuinely spiral
# into the origin instead of silently escaping to infinity.
for x0 in ([1.5, 0], [-1.2, 0.8], [0, -1.5]):
    sol = solve_ivp(rhs, [0, 30], x0, args=(mu,), max_step=0.02)
    ax.plot(sol.y[0], sol.y[1], color="C0", lw=1.3)

# The unstable limit cycle itself: integrate backwards in time from a point
# near the origin. Forward in time the origin is attracting, so backward in
# time the trajectory is repelled outward and converges onto the cycle.
lc = solve_ivp(rhs, [0, -60], [0.1, 0.0], args=(mu,), max_step=0.02)
tail = lc.t < -50
ax.plot(lc.y[0, tail], lc.y[1, tail], "k", lw=2.0, label="unstable limit cycle")

ax.plot(0, 0, "ko", ms=6)
ax.annotate("locally stable spiral", (0, 0), xytext=(0.15, 2.15), fontsize=10)
ax.annotate("unstable limit cycle\n(amplitude " + r"$\approx2.01$)", (2.01, 0),
            xytext=(0.9, -2.35), fontsize=9, arrowprops=dict(arrowstyle="->", lw=0.8))
ax.axhline(0, lw=0.5, c="k")
ax.axvline(0, lw=0.5, c="k")
ax.set_xlabel("x")
ax.set_ylabel("v")
ax.set_title(r"Problem 3(c): $\mu=-1$")
ax.set_xlim(-3, 3)
ax.set_ylim(-3, 3)
ax.legend(loc="upper left", fontsize=8)
savefig(fig, "Problem3c.png")
plt.close(fig)


mu = 1.0
X, V = np.meshgrid(np.linspace(-3, 3, 400), np.linspace(-3, 3, 400))
dX = V
dV = -X + mu * (1 - X**2) * V

fig, ax = plt.subplots(figsize=(6, 5))
ax.streamplot(X, V, dX, dV, density=1.4, color="gray", linewidth=0.7)

sol_in = solve_ivp(rhs, [0, 30], [0.1, 0.1], args=(mu,), max_step=0.02)
ax.plot(sol_in.y[0], sol_in.y[1], color="C0", lw=1.2, label="spirals out from near origin")

sol_out = solve_ivp(rhs, [0, 30], [3.0, 3.0], args=(mu,), max_step=0.02)
ax.plot(sol_out.y[0], sol_out.y[1], color="C1", lw=1.0, alpha=0.6, label="spirals in from outside")

tail = sol_out.t > 20
ax.plot(sol_out.y[0, tail], sol_out.y[1, tail], color="k", lw=2.0, label="limit cycle")

ax.plot(0, 0, "wo", ms=6, mec="k", mew=1.3)
ax.annotate("unstable spiral", (0, 0), xytext=(0.15, 2.6), fontsize=10)
ax.axhline(0, lw=0.5, c="k")
ax.axvline(0, lw=0.5, c="k")
ax.set_xlabel("x")
ax.set_ylabel("v")
ax.set_title(r"Problem 3(g): $\mu=+1$")
ax.set_xlim(-3, 3)
ax.set_ylim(-3, 3)
ax.legend(loc="upper left", fontsize=8)
savefig(fig, "Problem3g.png")
plt.close(fig)

print(f"Saved Problem3c/3g.png to: {os.path.abspath(FIG_DIR)}")
