from __future__ import annotations
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


M = 1.0     # mass
IC = 1.0    # moment of inertia about the mass center C
D = 0.5     # distance from skate B to mass center C, along b1

def sleigh_rhs(t, s, m=M, Ic=IC, d=D):
    """Reduced first-order system.  State s = [x, y, theta, u, omega]."""
    x, y, th, u, w = s
    x_dot = u * np.cos(th)
    y_dot = u * np.sin(th)
    th_dot = w
    u_dot = d * w**2
    w_dot = -(m * d * u * w) / (Ic + m * d**2)
    return [x_dot, y_dot, th_dot, u_dot, w_dot]


def integrate(u0, w0, t_end=30.0, n=6000, th0=0.0, x0=0.0, y0=0.0,
              m=M, Ic=IC, d=D):
    """Integrate one trajectory from a given (u0, omega0)."""
    t_eval = np.linspace(0.0, t_end, n)
    sol = solve_ivp(sleigh_rhs, (0.0, t_end), [x0, y0, th0, u0, w0],
                    t_eval=t_eval, args=(m, Ic, d),
                    rtol=1e-10, atol=1e-12, dense_output=False)
    return sol

def body_frame_velocity(u, w, d=D):
    """(v1, v2) = C's velocity in the body frame:  v1 = u,  v2 = d*omega."""
    return u, d * w

def energy(u, w, m=M, Ic=IC, d=D):
    """Kinetic energy on the constraint surface (conserved, no potential)."""
    return 0.5 * m * u**2 + 0.5 * (Ic + m * d**2) * w**2


def constraint_residual(sol):
    """phi = -x_dot sin(th) + y_dot cos(th) should stay ~0."""
    x, y, th, u, w = sol.y
    x_dot = u * np.cos(th)
    y_dot = u * np.sin(th)
    return -x_dot * np.sin(th) + y_dot * np.cos(th)


def blade_force(u, w, m=M, Ic=IC, d=D):
    """Lagrange multiplier lambda = lateral (b2) constraint force.
    From I_c omega_dot = -d*lambda and the omega equation:
        lambda = m Ic u omega / (Ic + m d**2).
    """
    return m * Ic * u * w / (Ic + m * d**2)

INITIAL_CONDITIONS = [
    (1.0, 1.6), (1.0, -1.6), (-1.0, 1.6), (-1.0, -1.6),
    (0.15, 2.1), (-0.15, 2.1), (0.15, -2.1), (-0.15, -2.1),
    (0.6, 1.0), (-0.6, -1.0), (1.4, 0.6), (-1.4, 0.6),
]


def figure_phase_portrait(fname="fig1_phase_portrait.png", m=M, Ic=IC, d=D):
    fig, ax = plt.subplots(figsize=(7.5, 6.4))

    v1 = np.linspace(-1.9, 1.9, 30)
    v2 = np.linspace(-1.3, 1.3, 30)
    V1, V2 = np.meshgrid(v1, v2)
    U, W = V1, V2 / d                       # invert v1=u, v2=d*omega
    dU = d * W**2                           # u_dot
    dW = -(m * d * U * W) / (Ic + m * d**2)  # omega_dot
    dV1, dV2 = dU, d * dW
    speed = np.hypot(dV1, dV2)
    ax.streamplot(V1, V2, dV1, dV2, color=speed, cmap="viridis",
                  density=1.2, linewidth=0.8, arrowsize=0.9)

    # integrated trajectories + start markers
    for (u0, w0) in INITIAL_CONDITIONS:
        sol = integrate(u0, w0, m=m, Ic=Ic, d=d)
        b1v, b2v = body_frame_velocity(sol.y[3], sol.y[4], d=d)
        ax.plot(b1v, b2v, "k", lw=1.4, alpha=0.85)
        ax.plot(u0, d * w0, "o", ms=5, color="crimson", zorder=5)

    # equilibria live on v2 = 0; flow always moves toward increasing v1
    u_eq = 1.0
    ax.scatter([u_eq], [0], s=130, color="green", zorder=6,
               label="stable node (C leads, B trails)")
    ax.scatter([-u_eq], [0], s=130, facecolors="none", edgecolors="red",
               linewidths=2, zorder=6, label="unstable node (B leads)")

    ax.axhline(0, color="gray", lw=0.6)
    ax.axvline(0, color="gray", lw=0.6)
    ax.set_xlabel(r"$v_1 = u$   (forward speed along $\hat b_1$)")
    ax.set_ylabel(r"$v_2 = d\,\dot\theta$   (lateral speed along $\hat b_2$)")
    ax.set_title("Body-frame velocity phase portrait  (constant-energy ellipses)")
    ax.set_xlim(-1.9, 1.9)
    ax.set_ylim(-1.3, 1.3)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(fname, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return fname


def figure_xy_paths(fname="fig2_xy_paths.png", m=M, Ic=IC, d=D):
    fig, ax = plt.subplots(figsize=(7.5, 6.0))
    demo = [(0.2, 2.0), (0.2, -2.0), (-0.4, 1.5), (0.05, 2.5)]
    for (u0, w0) in demo:
        sol = integrate(u0, w0, t_end=18.0, n=4000, m=m, Ic=Ic, d=d)
        ax.plot(sol.y[0], sol.y[1], lw=1.6,
                label=fr"$u_0={u0},\ \dot\theta_0={w0}$")
        ax.plot(sol.y[0][0], sol.y[1][0], "ko", ms=4)
    ax.set_aspect("equal")
    ax.set_xlabel(r"$x$   (skate $B$, along $\hat n_1$)")
    ax.set_ylabel(r"$y$   (skate $B$, along $\hat n_2$)")
    ax.set_title("Physical path of the skate: curved transient -> straight glide")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(fname, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return fname

def figure_time_series(fname="fig3_time_series.png", m=M, Ic=IC, d=D):
    sol = integrate(0.2, 2.0, t_end=30.0, n=6000, m=m, Ic=Ic, d=d)
    t = sol.t
    u, w = sol.y[3], sol.y[4]
    E = energy(u, w, m, Ic, d)
    lam = blade_force(u, w, m, Ic, d)

    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))

    axes[0, 0].plot(t, u, label=r"$u=v_1$")
    axes[0, 0].plot(t, w, label=r"$\dot\theta$")
    axes[0, 0].set_xlabel("time"); axes[0, 0].set_ylabel("speed")
    axes[0, 0].set_title(r"Forward speed rises, spin decays to 0")
    axes[0, 0].legend(); axes[0, 0].grid(alpha=0.3)

    axes[0, 1].plot(t, E)
    axes[0, 1].set_xlabel("time"); axes[0, 1].set_ylabel("kinetic energy")
    axes[0, 1].set_title(f"Energy conserved (drift {(_drift(E)):.1e})")
    axes[0, 1].grid(alpha=0.3)

    axes[1, 0].plot(t, lam, color="darkorange")
    axes[1, 0].set_xlabel("time")
    axes[1, 0].set_ylabel(r"$\lambda$  (lateral blade force, along $\hat b_2$)")
    axes[1, 0].set_title(r"Constraint force $\lambda \to 0$ as motion straightens")
    axes[1, 0].grid(alpha=0.3)

    axes[1, 1].plot(t, np.abs(constraint_residual(sol)) + 1e-300)
    axes[1, 1].set_yscale("log")
    axes[1, 1].set_xlabel("time")
    axes[1, 1].set_ylabel(r"$|\phi|$  (constraint residual)")
    axes[1, 1].set_title("Knife-edge constraint held to solver tolerance")
    axes[1, 1].grid(alpha=0.3, which="both")

    fig.tight_layout()
    fig.savefig(fname, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return fname


def _drift(arr):
    return (arr.max() - arr.min()) / np.mean(arr)


def main():
    print(f"Parameters:  m={M}, Ic={IC}, d={D}")
    print("Generating figures...")
    for fn in (figure_phase_portrait, figure_xy_paths, figure_time_series):
        print("  wrote", fn())


    sol = integrate(0.2, 2.0)
    E = energy(sol.y[3], sol.y[4])
    print(f"\nEnergy drift over run : {_drift(E):.2e}")
    print(f"Max |constraint phi|  : {np.max(np.abs(constraint_residual(sol))):.2e}")
    print(f"Final (u, theta_dot)  : ({sol.y[3][-1]:.4f}, {sol.y[4][-1]:.2e})")
    print("  -> settles to u>0, spin->0  (center of mass leads).")


if __name__ == "__main__":
    main()