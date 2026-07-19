import numpy as np
import matplotlib.pyplot as plt

theta_dot = 5   # theta_dot (rad/s), assumed constant
r = 1           # crank length MA (m)
h = 10          # height of M above O (m)
H = 20          # height of slot (B) above O (m)

theta_span = (0, 2 * np.pi)
theta_eval = np.linspace(theta_span[0], theta_span[-1], 500)

x_points = (H * r * np.cos(theta_eval)) / (h - r * np.sin(theta_eval))
x_dot_points = (H * r * (r - h * np.sin(theta_eval)) * theta_dot) / (h - r * np.sin(theta_eval)) ** 2

fig1, ax_main = plt.subplots(figsize=(8, 5.5))

ax_main.plot(x_points, x_dot_points)

ax_main.set_xlabel('x (m)')
ax_main.set_ylabel(r'$\dot{x}$ (m/s)')
ax_main.set_title(r'$\dot{x}$ vs $x$')
ax_main.grid(True, alpha=0.3)

fig2, (ax_x, ax_xdot) = plt.subplots(1, 2, figsize=(13, 5.5))

ax_x.plot(theta_eval, x_points)
ax_x.set_xlabel(r'$\theta$ (rad)')
ax_x.set_ylabel('x (m)')
ax_x.set_title(r'$x$ vs $\theta$')
ax_x.grid(True, alpha=0.3)

ax_xdot.plot(theta_eval, x_dot_points)
ax_xdot.set_xlabel(r'$\theta$ (rad)')
ax_xdot.set_ylabel(r'$\dot{x}$ (m/s)')
ax_xdot.set_title(r'$\dot{x}$ vs $\theta$')
ax_xdot.grid(True, alpha=0.3)

plt.show()
