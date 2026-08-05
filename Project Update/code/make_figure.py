"""
Schematic figure of the spring-loaded inverted pendulum (SLIP) model:
left panel shows the stance phase (spring-loaded pendulum in polar
coordinates r, theta), right panel shows the flight phase (ballistic
projectile). Saved to ../latex/Figures/progress_figure.png for use in
the project update.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

fig, axes = plt.subplots(1, 2, figsize=(8, 3.6))

# ---------------------------------------------------------------
# Left panel: stance phase (spring-loaded pendulum, polar coords)
# ---------------------------------------------------------------
ax = axes[0]
ax.axhline(0, color="k", linewidth=1.5)  # ground

foot = np.array([0.0, 0.0])
theta = np.deg2rad(70)   # leg angle from ground
r = 1.0                  # leg length
mass = foot + r * np.array([np.cos(theta), np.sin(theta)])

# zigzag "spring" leg from foot to mass
n_zig = 7
t = np.linspace(0, 1, 2 * n_zig + 1)
amp = 0.05
perp = np.array([-np.sin(theta), np.cos(theta)])
zig = np.array([
    foot + tt * (mass - foot) + ((-1) ** i) * amp * perp if 0 < i < len(t) - 1 else foot + tt * (mass - foot)
    for i, tt in enumerate(t)
])
ax.plot(zig[:, 0], zig[:, 1], color="tab:blue", linewidth=1.8)

# point mass
ax.add_patch(patches.Circle(mass, 0.09, color="tab:red", zorder=5))
ax.plot(*foot, marker="^", color="k", markersize=10, zorder=5)

# angle arc for theta
arc = patches.Arc(foot, 0.5, 0.5, angle=0, theta1=0, theta2=np.degrees(theta),
                   color="gray", linewidth=1.2)
ax.add_patch(arc)
ax.text(0.32, 0.10, r"$\theta$", fontsize=11, color="gray")

# r label
mid = foot + 0.55 * (mass - foot)
ax.text(mid[0] + 0.06, mid[1], r"$r$", fontsize=12, color="tab:blue")

ax.set_xlim(-0.3, 1.1)
ax.set_ylim(-0.15, 1.15)
ax.set_aspect("equal")
ax.axis("off")
ax.set_title("Stance phase\n(spring-loaded pendulum)", fontsize=10)

# ---------------------------------------------------------------
# Right panel: flight phase (ballistic projectile)
# ---------------------------------------------------------------
ax = axes[1]
ax.axhline(0, color="k", linewidth=1.5)  # ground

x = np.linspace(0, 1.4, 200)
y0, vx, vy0, g = 0.0, 1.0, 1.1, 2.2
tt = x / vx
y = y0 + vy0 * tt - 0.5 * g * tt ** 2
mask = y >= 0
ax.plot(x[mask], y[mask], "--", color="tab:blue", linewidth=1.8)

apex_idx = np.argmax(y)
ax.add_patch(patches.Circle((x[apex_idx], y[apex_idx]), 0.05, color="tab:red", zorder=5))
ax.add_patch(patches.Circle((x[0], y[0]), 0.05, color="tab:red", alpha=0.4, zorder=5))
last = np.where(mask)[0][-1]
ax.add_patch(patches.Circle((x[last], y[last]), 0.05, color="tab:red", alpha=0.4, zorder=5))

ax.annotate("", xy=(x[apex_idx] + 0.18, y[apex_idx]), xytext=(x[apex_idx], y[apex_idx]),
            arrowprops=dict(arrowstyle="->", color="gray"))
ax.text(x[apex_idx] + 0.20, y[apex_idx] - 0.02, r"$v_x$", fontsize=10, color="gray")

ax.set_xlim(-0.1, 1.5)
ax.set_ylim(-0.15, 0.9)
ax.set_aspect("equal")
ax.axis("off")
ax.set_title("Flight phase\n(ballistic projectile)", fontsize=10)

fig.suptitle("SLIP model: two-phase hybrid dynamics", fontsize=12, y=1.02)
fig.tight_layout()

out_dir = os.path.join("..", "latex", "Figures")
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "progress_figure.png")
fig.savefig(out_path, dpi=300, bbox_inches="tight")
print(f"Saved figure to {out_path}")
