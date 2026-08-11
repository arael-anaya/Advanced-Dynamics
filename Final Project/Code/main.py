import copy
import time
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
    y_TD_dot = - np.sqrt(2*g*(y_o - r_o * np.sin(theta_TD)))

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
    fig.savefig(os.path.join(SCRIPT_DIR, 'stance_validation.png'), dpi=200)
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
    x_dot_pred = np.sqrt(2*E1/ params['m'] - 2*params['g']*finalState[1])
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

    x_dot = np.sqrt(2*E/ params['m'] - 2*params['g']*y)

    inputApexState = [0, y , x_dot , 0]
    outputApexState = hop(inputApexState, thetaTD, params)

    return outputApexState[1]

# ---------- find the fixed point of the apex map and its local stability ----------

def findApexMapFixedPoint(params, thetaTD, y_o=y_o, v_o=v_o):
    initialState = [0,y_o,v_o,0]
    E = energyHop(initialState, params)
    f = lambda y: apexmap(y, thetaTD, params , E) - y

    # +.001 avoids singularities 
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

    y_star: float = brentq(f, y_min, y_max)  # type: ignore[assignment]

    dy = max(1e-4 * abs(y_star), 1e-6)

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
    n_failed = 0
    n_total = len(k_values) * len(thetaTD_values)

    # High-resolution sweeps can take well over an hour unattended, so print
    # a per-row progress/ETA line rather than going silent until the end.
    t_start = time.time()

    for i, k_val in enumerate(k_values):
        for j, theta_val in enumerate(thetaTD_values):
            trial_params = copy.deepcopy(base_params)
            trial_params['k'] = k_val
            try:
                y_star, slope = findApexMapFixedPoint(trial_params, theta_val, y_o, v_o)
            except ValueError:
                n_failed +=1
                continue
            y_star_grid[i, j] = y_star
            slope_grid[i, j] = slope

        n_done = (i + 1) * len(thetaTD_values)
        elapsed = time.time() - t_start
        eta = elapsed * (n_total - n_done) / n_done
        print(f"  row {i+1}/{len(k_values)} (k={k_val:.0f}) done -- "
              f"{n_done}/{n_total} cells, {elapsed/60:.1f} min elapsed, "
              f"~{eta/60:.1f} min remaining")

    print(f"{n_failed}/{n_total} cells failed to find a fixed point")
    return y_star_grid, slope_grid



import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

def plotStabilitySweep(k_values, thetaTD_values, y_star_grid, slope_grid,
                        friction_floor_deg=59.0, save_path=None):
    """
    Classifies each grid cell as stable / unstable / no fixed point,
    overlays the friction-cone floor, and saves the deliverable figure.
    """
    K, THETA = np.meshgrid(k_values, thetaTD_values, indexing='ij')

    # 0 = no fixed point, 1 = unstable, 2 = stable
    classification = np.zeros_like(slope_grid)
    found = ~np.isnan(slope_grid)
    classification[found & (np.abs(slope_grid) < 1)] = 2
    classification[found & (np.abs(slope_grid) >= 1)] = 1

    cmap = ListedColormap(['#d9d9d9', '#e07b54', '#4c9a6a'])  # gray, orange, green
    bounds = [-0.5, 0.5, 1.5, 2.5]
    norm = BoundaryNorm(bounds, cmap.N)

    fig, ax = plt.subplots(figsize=(8, 6))
    mesh = ax.pcolormesh(THETA, K, classification, cmap=cmap, norm=norm, shading='auto')

    # friction-cone floor
    ax.axvline(friction_floor_deg, color='k', linestyle='--', linewidth=1.5)
    ax.axvspan(thetaTD_values.min(), friction_floor_deg, color='k', alpha=0.08)
    ax.text(friction_floor_deg + 0.5, k_values.max() * 0.97,
            r'$\theta_{TD} \geq %.0f^\circ$ (friction cone, $\mu$=0.6)' % friction_floor_deg,
            va='top', fontsize=9)

    ax.set_xlabel(r'Touchdown angle $\theta_{TD}$ (deg)')
    ax.set_ylabel(r'Leg stiffness $k$')
    ax.set_title('Self-stable running region')

    cbar = fig.colorbar(mesh, ax=ax, ticks=[0, 1, 2])
    cbar.ax.set_yticklabels(['No fixed point', 'Unstable', 'Stable'])

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300)
    return fig, ax


# ============================================================
# Stage 7 - Local refinement: 
# ============================================================

def localRefinementSweep(base_params, k_center, theta_center_deg,
                          k_half_width=2500.0, theta_half_width_deg=2.5,
                          n=20, y_o=y_o, v_o=v_o):
    """Rerun a small (k, thetaTD) window at finer resolution, zoomed on a
    stable cell found by the coarse sweep."""
    k_values = np.linspace(k_center - k_half_width, k_center + k_half_width, n)
    thetaTD_deg = np.linspace(theta_center_deg - theta_half_width_deg,
                               theta_center_deg + theta_half_width_deg, n)
    thetaTD_values = np.radians(thetaTD_deg)

    y_star_grid, slope_grid = gridSearchFixedPoints(base_params, k_values, thetaTD_values, y_o, v_o)
    return k_values, thetaTD_deg, y_star_grid, slope_grid


def stableBandWidth(slope_grid):
    """Longest run of contiguous stable (|slope| < 1) cells found along any
    single row (fixed k, varying thetaTD) or column (fixed thetaTD, varying
    k). A width of 1 on a fine grid means the band really is that thin at
    this resolution; >1 means the coarse sweep's one-cell staircase was an
    artifact of the coarse spacing."""
    stable = (~np.isnan(slope_grid)) & (np.abs(slope_grid) < 1)

    def longest_run(mask_1d):
        best = run = 0
        for is_stable in mask_1d:
            run = run + 1 if is_stable else 0
            best = max(best, run)
        return best

    row_runs = [longest_run(row) for row in stable]
    col_runs = [longest_run(col) for col in stable.T]
    return max(row_runs, default=0), max(col_runs, default=0)


# ============================================================
# Run
# ============================================================

import os
import warnings

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

RESULTS_PATH = os.path.join(SCRIPT_DIR, "sweep_results.npz")
RERUN_SWEEP = False  # set True to force a fresh sweep even if a cached file exists


def runOrLoadSweep(base_params, k_values, thetaTD_values, y_o, v_o,
                    results_path=RESULTS_PATH, rerun=RERUN_SWEEP):
    if not rerun and os.path.exists(results_path):
        data = np.load(results_path)
        cached_k = data['k_values']
        cached_theta = data['thetaTD_values']
        if (cached_k.shape == k_values.shape and np.allclose(cached_k, k_values)
                and cached_theta.shape == thetaTD_values.shape
                and np.allclose(cached_theta, thetaTD_values)):
            print(f"Loaded cached sweep from {results_path}")
            return data['y_star_grid'], data['slope_grid']
        else:
            print("Cached grid does not match requested k_values/thetaTD_values - rerunning.")

    print("Running fresh sweep...")
    y_star_grid, slope_grid = gridSearchFixedPoints(base_params, k_values, thetaTD_values, y_o, v_o)
    np.savez(results_path,
             k_values=k_values,
             thetaTD_values=thetaTD_values,
             y_star_grid=y_star_grid,
             slope_grid=slope_grid)
    print(f"Saved sweep to {results_path}")
    return y_star_grid, slope_grid

if __name__ == "__main__":
    warnings.filterwarnings('error', category=np.exceptions.ComplexWarning)

    # Bumped from 40x40: the local refinement check below showed the coarse
    # grid's one-cell-wide stable "staircase" was a resolution artifact, not
    # a real measure-zero band -- it thickened into a connected patch at 4x
    # finer spacing. This resolution matches that refinement window's cell
    # spacing (~0.26 deg, ~265 stiffness) across the full sweep range.
    # ~12,600 cells; at ~0.47 s/cell (measured on this machine) that's
    # roughly 90-100 minutes. Grab coffee.
    k_sweep = np.linspace(5000, 40000, 133)
    thetaTD_sweep_deg = np.linspace(55, 80, 95)
    thetaTD_sweep_rad = np.radians(thetaTD_sweep_deg)

    y_star_grid, slope_grid = runOrLoadSweep(params, k_sweep, thetaTD_sweep_rad, y_o, v_o)
    print("y_star grid:\n", y_star_grid)
    print("slope grid:\n", slope_grid)

    fig, ax = plotStabilitySweep(k_sweep, thetaTD_sweep_deg, y_star_grid, slope_grid,
                                  friction_floor_deg=59.0,
                                  save_path=os.path.join(SCRIPT_DIR, 'stability_sweep.png'))

    coarse_row_run, coarse_col_run = stableBandWidth(slope_grid)
    print(f"coarse grid stable band width: {coarse_row_run} cell(s) along theta, "
          f"{coarse_col_run} cell(s) along k")

    # ---- local refinement: zoom a 20x20 window (5 deg x 5000 stiffness,
    # matching one dip below the friction-cone floor at 59 deg so we're not
    # zooming into a physically-excluded corner) onto a stable cell from the
    # coarse sweep and see if the band thickens at 4x the resolution ----
    stable_coarse = (~np.isnan(slope_grid)) & (np.abs(slope_grid) < 1)
    stable_idx = np.argwhere(stable_coarse)
    if stable_idx.size == 0:
        print("no stable cells found in the coarse sweep -- skipping refinement")
    else:
        i, j = stable_idx[len(stable_idx) // 2]  # a representative stable cell, not an edge case
        k_center = k_sweep[i]
        theta_center_deg = thetaTD_sweep_deg[j]
        print(f"refining around k={k_center:.1f}, thetaTD={theta_center_deg:.2f} deg")

        k_fine, theta_fine_deg, y_star_fine, slope_fine = localRefinementSweep(
            params, k_center, theta_center_deg,
            k_half_width=2500.0, theta_half_width_deg=2.5, n=20, y_o=y_o, v_o=v_o)

        fine_row_run, fine_col_run = stableBandWidth(slope_fine)
        print(f"refined grid stable band width: {fine_row_run} cell(s) along theta, "
              f"{fine_col_run} cell(s) along k (out of {len(theta_fine_deg)}x{len(k_fine)} cells, "
              f"~{(theta_fine_deg[1]-theta_fine_deg[0]):.3f} deg / {(k_fine[1]-k_fine[0]):.1f} stiffness spacing)")

        if fine_row_run <= 1 and fine_col_run <= 1:
            print("band stays one-cell-wide at 4x finer resolution -- looks like a real, "
                  "genuinely narrow stability margin, not a resolution artifact.")
        else:
            print("band thickens at finer resolution -- the coarse sweep's staircase was "
                  "a grid artifact; bump the full sweep's resolution before finalizing.")

        plotStabilitySweep(k_fine, theta_fine_deg, y_star_fine, slope_fine,
                            friction_floor_deg=59.0,
                            save_path=os.path.join(SCRIPT_DIR, 'stability_refinement.png'))