import copy
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares
from scipy.optimize import brentq

# Paramerts
m = 80
r_o = 1
k = 20000
g = 9.81
thetaTD = np.radians(68)
y_o = 1.2
v_o = 3.0

params = {'m' : m , 'k' : k , 'g' : g, 'r_o' : r_o , 'y_o' : y_o}


# ============================================================
# Stage 1 - Stance phase: touchdown -> standing -> liftoff
# First piece of the model: integrate a single stance phase and
# check that total mechanical energy is conserved through it.
# ============================================================

def touchdownState(x_o_dot , y_o, theta_TD, r_o, g):
    r = r_o
    theta = np.arctan2(r_o * np.sin(theta_TD) , -r_o * np.cos(theta_TD))

    x_TD_dot = x_o_dot
    y_TD_dot = - (2*g*(y_o - r_o * np.sin(theta_TD)))**.5

    r_dot = x_TD_dot * np.cos(theta) + y_TD_dot * np.sin(theta)

    theta_dot = (-x_TD_dot * np.sin(theta) + y_TD_dot * np.cos(theta)) / r_o

    return [r , theta, r_dot , theta_dot]

def standingState(t , s, params):
    r_dot = s[2]
    theta_dot = s[3]

    r_ddot = s[0] * theta_dot**2 - params['g'] * np.sin(s[1]) - (params['k'] / params['m'])* (s[0] - params['r_o'])
    theta_ddot = -(2/s[0]) * r_dot * theta_dot - (params['g'] / s[0]) * np.cos(s[1])

    return [r_dot , theta_dot, r_ddot, theta_ddot]

def liftoffStateSwitch(t, s, params):
    return params['k'] * (s[0] - params['r_o'])
liftoffStateSwitch.terminal = True
liftoffStateSwitch.direction = 1

def energy(s, params):
    E = (1/2) * params['m'] * (s[2]**2 + s[0]**2 * s[3]**2) + params['m'] * params['g'] * s[0] * np.sin(s[1]) + (1/2) * params['k']* (s[0] - params['r_o'])**2

    return E

# ---------- run a single stance phase for diagnostics ----------
# (mirrors the body of hop(), but keeps sol/s0 around for plotting)

def initialFunctionTest():
    s0 = touchdownState(v_o, y_o, thetaTD, r_o, g)

    t_span = (0.0, 1.0)
    sol = solve_ivp(standingState, t_span, s0, args=(params,),
                    events=liftoffStateSwitch, rtol=1e-10, atol=1e-12,
                    dense_output=True, max_step=1e-3)

    E = np.array([energy(sol.y[:, i], params) for i in range(sol.y.shape[1])])
    drift = np.max(np.abs(E - E[0])) / np.abs(E[0])

    # Plotting and data stuff, AI can fill in these gaps as simply learning to plot good is not hard and not my current goal

    print("status:", sol.status)
    print("liftoffStateSwitch time:", sol.t_events[0])
    print("drift:", drift)

    # ---------- diagnostics ----------
    r_TD, th_TD, rdot_TD, thdot_TD = s0
    s_LO = sol.y_events[0][0]
    t_LO = sol.t_events[0][0]

    print("--- touchdown state ---")
    print(f"  r      = {r_TD:.6f} m")
    print(f"  theta  = {np.degrees(th_TD):.4f} deg")
    print(f"  r_dot  = {rdot_TD:.6f} m/s")
    print(f"  th_dot = {thdot_TD:.6f} rad/s")

    print("--- liftoffStateSwitch state ---")
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
    ax[1, 0].plot(x_com[-1], y_com[-1], 's', color='C2', label='liftoffStateSwitch')
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


# ============================================================
# Stage 2 - Apex state: ballistic flight after liftoff
# Convert the liftoff (r, theta) state back to Cartesian (x, y)
# and propagate the free-flight trajectory forward to the apex
# (the point where the vertical velocity goes to zero).
# ============================================================

def projectileApexState(s):
    r_o = s[0]
    theta_o = s[1]
    r_o_dot = s[2]
    theta_o_dot = s[3]

    # convert state from liftoff back to cartesian
    y_o = r_o * np.sin(theta_o)
    x_o = r_o * np.cos(theta_o)

    y_o_dot = r_o_dot * np.sin(theta_o) + theta_o_dot * r_o * np.cos(theta_o)
    x_o_dot = r_o_dot * np.cos(theta_o) - theta_o_dot * r_o * np.sin(theta_o)

    # Solve for when y velocity = 0, this is the highest point
    t_apex = y_o_dot / g

    # find corresponding x y x_dot y_dot values
    y_dot = -g*t_apex + y_o_dot
    x_dot = x_o_dot

    y = y_o - ((g/2) * t_apex**2) + y_o_dot * t_apex
    x = x_o + x_dot * t_apex

    return [x , y, x_dot , y_dot]


# ============================================================
# Stage 3 - Hop: touchdown -> stance -> liftoff -> apex
# Chains stages 1-2 together into a single map that takes an
# apex state and returns the next apex state, one hop later.
# ============================================================

def hop(apexState ,thetaTD, params):
    s0 = touchdownState(apexState[2],apexState[1], thetaTD, params['r_o'], params['g'])
    t_span = (0.0,1.0)

    sol = solve_ivp(standingState, t_span, s0, args=(params,),
                events=liftoffStateSwitch, rtol=1e-10, atol=1e-12,
                dense_output=True, max_step=1e-3)

    s1 = projectileApexState(sol.y_events[0][0])

    return s1

def energyHop(apexState, params):
    E = (1/2) * params['m'] * (apexState[2]**2 + apexState[3]**2) + params['m'] * params['g'] * apexState[1]
    return E


# ============================================================
# Stage 4 - x is redundant: the apex-to-apex map is fully
# determined by total energy E and height y, not by x directly.
# ============================================================

def proveX_dotIsRedundant():
    # E = (1/2)*m*(x_dot^2 + y_dot^2) + m*g*y
    # ((E - mgy) * 2/m - y_dot**2))**.5 = x_dot
    # Y_dot = 0
    # ((E-mgy)*2/m)**.5 = x_dot
    # (2E/m - 2gy)**.5 = x_dot

    initialState = [0,y_o,v_o,0]
    initialThetaTD = thetaTD

    E1 = energyHop(initialState , params)


    finalState = hop(initialState, initialThetaTD, params)
    x_dot_pred = (2*E1/ params['m'] - 2*params['g']*finalState[1])**.5
    E2 = energyHop(finalState, params)

    print(f"Energy Drift over 1 hop: {E2-E1}")
    print(f"X_dot Drift over 1 hop: {finalState[2]-x_dot_pred}")

# ============================================================
# Stage 5 - Reduced 1-D apex map: fixed point and stability
# Stage 4 showed the hop only depends on E and y, so collapse the
# hop into a 1-D map y -> apexmap(y) and find the periodic apex
# height (fixed point), then check its local stability from the
# map's slope there.
# ============================================================

def apexmap(y , thetaTD, params, E):
    y = np.asarray(y).item()

    x_dot = (2*E/ params['m'] - 2*params['g']*y)**.5

    inputApexState = [0, y , x_dot , 0]
    outputApexState = hop(inputApexState, thetaTD, params)

    return outputApexState[1]

# ---------- find the fixed point of the apex map and its local stability ----------

def findApexMapFixedPoint(params, thetaTD, y_o=y_o, v_o=v_o):
    initialState = [0,y_o,v_o,0]
    E = energyHop(initialState, params)
    f = lambda y: apexmap(y, thetaTD, params , E) - y

    y_min = params['r_o'] * np.sin(thetaTD) + .001 # apex must be at/above touchdown height for this thetaTD
    # 2E/m - 2*g*y >= 0
    #  Y < 2E/ (2mg)
    y_max = 2*E / (2* params['m'] * params['g'])

    y_sample = np.linspace(y_min, y_max, 30)
    f_sample = np.array([f(yi) for yi in y_sample])

    sign_changes = np.where(np.diff(np.sign(f_sample)) != 0)[0]

    if sign_changes.size == 0:
        raise ValueError(f"no sign change in apex map for thetaTD={thetaTD}, k={params['k']}")

    y_min = y_sample[sign_changes[0]]
    y_max = y_sample[sign_changes[0] + 1]

    y_star = brentq(f, y_min, y_max)
    dy = .001

    apexmapPrime = (apexmap(y_star + dy, thetaTD, params, E) - apexmap(y_star - dy, thetaTD, params, E)) / (2 * dy) # type: ignore

    # print(apexmapPrime)

    return (y_star, apexmapPrime)


# ============================================================
# Stage 6 - Grid search: fixed point and stability over (k, thetaTD)
# Each grid point needs its own params dict (deep-copied off the
# base params) so sweeping 'k' never mutates the shared dict that
# other stages/functions read from.
# ============================================================

def gridSearchFixedPoints(base_params, k_values, thetaTD_values, y_o=y_o, v_o=v_o):
    y_star_grid = np.full((len(k_values), len(thetaTD_values)), np.nan)
    slope_grid = np.full((len(k_values), len(thetaTD_values)), np.nan)

    for i, k_val in enumerate(k_values):
        for j, theta_val in enumerate(thetaTD_values):
            trial_params = copy.deepcopy(base_params)
            trial_params['k'] = k_val
            try:
                y_star, slope = findApexMapFixedPoint(trial_params, theta_val, y_o, v_o)
            except ValueError:
                continue
            y_star_grid[i, j] = y_star
            slope_grid[i, j] = slope

    return y_star_grid, slope_grid


# ============================================================
# Run
# ============================================================

if __name__ == "__main__":
#     initialFunctionTest()
#     proveX_dotIsRedundant()
    print(findApexMapFixedPoint(params, thetaTD))

    k_sweep = np.linspace(5000, 40000, 5)
    thetaTD_sweep = np.linspace(np.radians(50), np.radians(80), 5)
    y_star_grid, slope_grid = gridSearchFixedPoints(params, k_sweep, thetaTD_sweep)
    print("y_star grid:\n", y_star_grid)
    print("slope grid:\n", slope_grid)
