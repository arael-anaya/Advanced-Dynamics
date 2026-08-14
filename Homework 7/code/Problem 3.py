
import os
import numpy as np
import matplotlib.pyplot as plt

FIG_DIR = os.path.join(os.path.dirname(__file__), "..", "latex", "Figures", "Graphs")
os.makedirs(FIG_DIR, exist_ok=True)


def savefig(fig, name):
    fig.savefig(os.path.join(FIG_DIR, name), dpi=200, bbox_inches="tight")


L1 = 3
L2 = 2 
L3 = 1

def getBeta(x , y):
    num = L1**2 + L2**2 - x**2 - y**2
    den = 2 * L1 * L2

    beta = np.arccos(num/den)


    return np.array((beta , -beta))


def getAlpha(x , y):

    num =  x**2 + y**2 + L1**2 - L2**2 
    den = 2 * L1 * np.sqrt(x**2 + y**2)

    alpha = np.arccos(num/den)

    return np.array((alpha, -alpha))


def forward_kinematics(theta1, theta2, theta3):

    def transform(theta, L):
        return np.array([
            [np.cos(theta) , -np.sin(theta) , L*np.cos(theta)],
            [np.sin(theta) , np.cos(theta) , L*np.sin(theta)],
            [0  , 0 ,  1]
        ])

    T1 = transform(theta1 , L1)
    T2 = transform(theta2 , L2)
    T3 = transform(theta3 , L3)

    T = T1 @ T2 @ T3

    x = T[0,2]
    y = T[1,2]
    theta = theta1 + theta2 + theta3

    return np.array([x,y,theta])


def get_joint_positions(theta1, theta2, theta3):
    p0 = np.array([0.0, 0.0])
    p1 = p0 + L1 * np.array([np.cos(theta1), np.sin(theta1)])
    p2 = p1 + L2 * np.array([np.cos(theta1 + theta2), np.sin(theta1 + theta2)])
    p3 = p2 + L3 * np.array([np.cos(theta1 + theta2 + theta3), np.sin(theta1 + theta2 + theta3)])
    return np.array([p0, p1, p2, p3])


def add_angle_arc(ax, center, start_angle, end_angle, radius, label, color="#52514e"):
    # arc sweeping from the local horizontal (start_angle) to the link direction (end_angle)
    t = np.linspace(start_angle, end_angle, 60)
    ax.plot(center[0] + radius * np.cos(t), center[1] + radius * np.sin(t),
            color=color, lw=1.3, zorder=6)

    # dashed reference line = the "horizontal" this angle is measured from
    ref_len = radius * 1.6
    ax.plot([center[0], center[0] + ref_len * np.cos(start_angle)],
            [center[1], center[1] + ref_len * np.sin(start_angle)],
            ls="--", color=color, lw=1.0, alpha=0.7, zorder=2)

    mid = (start_angle + end_angle) / 2
    label_x = center[0] + radius * 1.35 * np.cos(mid)
    label_y = center[1] + radius * 1.35 * np.sin(mid)
    ax.plot(label_x, label_y, alpha=0)  # register label position for autoscaling
    ax.annotate(label, (label_x, label_y), ha="center", va="center",
                fontsize=9, color=color, zorder=6)


def plot_solution_with_angles(ax, theta1, theta2, theta3, x, y, theta, color):
    p0, p1, p2, p3 = get_joint_positions(theta1, theta2, theta3)

    ax.plot([p0[0], p1[0], p2[0], p3[0]], [p0[1], p1[1], p2[1], p3[1]],
            "-o", color=color, lw=2.5, ms=8, mfc="white", mec=color, mew=1.5, zorder=3)
    ax.plot(p0[0], p0[1], "s", color="#0b0b0b", ms=9, zorder=4)
    ax.plot(x, y, "k*", ms=14, zorder=5)
    ax.annotate("target", (x, y), textcoords="offset points", xytext=(8, 8),
                fontsize=8, color="#52514e")


    add_angle_arc(ax, p0, 0.0, theta1, 0.45, r"$\theta_1$", color)
    add_angle_arc(ax, p1, theta1, theta1 + theta2, 0.40, r"$\theta_2$", color)
    add_angle_arc(ax, p2, theta1 + theta2, theta1 + theta2 + theta3, 0.35, r"$\theta_3$", color)
    add_angle_arc(ax, p3, 0.0, theta, 0.55, r"$\theta$", "#0b0b0b")


def plot_state(name, x, y, theta, theta1_sols, theta2_sols, theta3_sols):
    colors = ["#2a78d6", "#eb6834"]
    n_sol = len(theta1_sols)

    fig, axes = plt.subplots(1, n_sol, figsize=(6 * n_sol, 6))
    if n_sol == 1:
        axes = [axes]

    for sol, ax in enumerate(axes):
        plot_solution_with_angles(ax, theta1_sols[sol], theta2_sols[sol], theta3_sols[sol],
                                    x, y, theta, colors[sol % len(colors)])
        ax.axhline(0, lw=0.5, c="k")
        ax.axvline(0, lw=0.5, c="k")
        ax.set_aspect("equal")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title(f"Solution {sol + 1}")
        ax.grid(True, lw=0.4, alpha=0.5)

    fig.suptitle(f"Problem 3: Planar 3R IK Solutions -- State {name}")
    fig.tight_layout()
    savefig(fig, f"Problem3_{name}.png")
    plt.close(fig)


def main():

    states = {'A' : [4, 0.5 , 22] , 'B' : [1.5 , -1.5, -105]}

    for name, state in states.items():
        print(f"State {name}: {state}")
        x = state[0]
        y = state[1]
        theta =  np.radians(state[2])

        x_w = x - L3 * np.cos(theta)
        y_w = y - L3 * np.sin(theta)

        gamma = np.arctan2(y_w, x_w)

        alpha = getAlpha(x_w, y_w)
        beta = getBeta(x_w , y_w)

        theta1 = gamma - alpha
        theta2 = np.pi - np.array(beta)
        theta3 = theta - theta1 - theta2

        for sol in range(len(theta1)):

            print(f"Solution {sol+1}:")
            print(f"Theta 1: {theta1[sol]}")
            print(f"Theta 2: {theta2[sol]}")
            print(f"Theta 3: {theta3[sol]}")

            check = forward_kinematics(theta1[sol], theta2[sol], theta3[sol])
            print(f"Solution Validation: {check}")

        plot_state(name, x, y, theta, theta1, theta2, theta3)



    


if __name__ == "__main__":
    main()

