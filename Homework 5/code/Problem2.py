import os
import numpy as np
import matplotlib.pyplot as plt

FIG_DIR = os.path.join(os.path.dirname(__file__), "..", "latex", "Figures", "Phase Portraits")
os.makedirs(FIG_DIR, exist_ok=True)


def savefig(fig, name):
    fig.savefig(os.path.join(FIG_DIR, name), dpi=200, bbox_inches="tight")


U, V = np.meshgrid(np.linspace(-1.2, 3.2, 400), np.linspace(-2.5, 2.5, 400))
dU = V
dV = U - 1.0

fig, ax = plt.subplots(figsize=(7, 5.5))
ax.streamplot(U, V, dU, dV, density=1.4, color="gray", linewidth=0.7)

ax.axvspan(-1.2, 0, color="0.85", zorder=0)
ax.annotate("nonphysical\n(u < 0)", (-0.6, -1.3), ha="center", fontsize=8, style="italic")
ax.axvline(0, lw=1.0, c="k")
ax.annotate("bottom of tube, r = 0", (0.05, -2.3), fontsize=8)

us = np.linspace(-1.2, 3.2, 50)
ax.plot(us, (us - 1), "--", color="C3", lw=1.5, label=r"unstable manifold $v=u-1$ ($\lambda=+1$)")
ax.plot(us, -(us - 1), "--", color="C0", lw=1.5, label=r"stable manifold $v=-(u-1)$ ($\lambda=-1$)")

ax.plot(1, 0, "ks", ms=7)
ax.annotate("saddle (1, 0)", (1, 0), xytext=(1.15, 0.25), fontsize=9)

u0, v0_crit = 2.0, -1.0
ax.plot(u0, v0_crit, "o", color="black", ms=6)
ax.annotate(r"$(u_0, v_0) = (2, -1)$" + "\ncritical inward speed", (u0, v0_crit),
            xytext=(1.9, -2.1), fontsize=8, arrowprops=dict(arrowstyle="->", lw=0.8))

t = np.linspace(0, 4, 400)
for v0, color, label in (
    (-1.3, "C1", "faster than critical: reaches r = 0"),
    (-1.0, "black", "exactly critical: asymptotes to saddle"),
    (-0.7, "C2", "slower than critical: turns back"),
):
    A = (u0 - 1 + v0) / 2
    B = (u0 - 1 - v0) / 2
    u_t = 1 + A * np.exp(t) + B * np.exp(-t)
    v_t = A * np.exp(t) - B * np.exp(-t)
    mask = (u_t >= -1.2) & (u_t <= 3.2) & (v_t >= -2.5) & (v_t <= 2.5)
    ax.plot(u_t[mask], v_t[mask], color=color, lw=1.6, label=label)

ax.set_xlabel("u")
ax.set_ylabel("v")
ax.set_title("Problem 2(c): non-dimensionalized rotating-bead system (saddle)")
ax.set_xlim(-1.2, 3.2)
ax.set_ylim(-2.5, 2.5)
ax.legend(loc="upper left", fontsize=7.5)
savefig(fig, "Problem2c.png")
plt.close(fig)

print(f"Saved Problem2c.png to: {os.path.abspath(FIG_DIR)}")
