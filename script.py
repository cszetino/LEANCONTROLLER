# script.py
# Problem 1: Lean-Into-Corner Seat Controller
# Enhancement: Map-curvature vs yaw-rate-only reference generation
# FIXED: Explicit occupant comfort metric (lateral specific force)
# ALIGNED: Seat-only tilt, mass cancels, friction excluded by assumption

import math
import time
import os
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# MODELING ASSUMPTIONS
# ============================================================
# 1) Vehicle does NOT tilt; the SEAT tilts
# 2) Occupant is rigidly attached to the seat (no slip)
# 3) Comfort is defined as zero lateral specific force
# 4) Occupant mass cancels from comfort equations
# 5) Seat-occupant friction is not explicitly modeled
# 6) Yaw-rate-only controller estimates lateral acceleration indirectly
# ============================================================

# ============================================================
# CONFIG
# ============================================================

T_SIM = 20.0
DT = 0.01

# Yaw-rate sensor model
YAW_BIAS_RAD_S = 0.0
YAW_NOISE_STD = 0.01
YAW_FILTER_TAU_S = 0.25

V_EPS = 1e-3
RNG_SEED = 1

# Output folder for plots (keeps your project root clean)
PLOTS_DIR = "plots"
SAVE_PNG = True       # set False if you don't want to save files
SHOW_WINDOWS = False  # set True if you want plot windows to pop up

# ============================================================
# PARAMETERS
# ============================================================

class LeanControllerParams:
    def __init__(self):
        self.g = 9.81

        # Seat roll limits
        self.alpha_max_deg = 15.0
        self.alpha_max_rad = math.radians(self.alpha_max_deg)

        # Seat roll dynamics (assumed)
        self.I = 30.0     # kg*m^2
        self.b = 8.0      # N*m*s/rad

        # PD gains (design parameters)
        self.Kp = 120.0
        self.Kd = 25.0

        # Speed tracking
        self.v_track_tau = 1.0

# ============================================================
# HELPERS
# ============================================================

def clamp(x, lo, hi):
    return lo if x < lo else (hi if x > hi else x)

def lowpass_1st_order(x_prev, x_meas, tau, dt):
    a = dt / max(tau, 1e-9)
    return x_prev + a * (x_meas - x_prev)

def speed_limit_for_curvature(kappa, alpha_max_rad, g):
    if abs(kappa) < 1e-9:
        return float("inf")
    ay_max = g * math.tan(alpha_max_rad)
    return math.sqrt(ay_max / abs(kappa))

def speed_dynamics_step(v, v_cmd, tau_v, dt):
    return v + (v_cmd - v) * (dt / max(tau_v, 1e-9))

def roll_dynamics_step(alpha, alpha_dot, tau, I, b, dt):
    alpha_ddot = (tau - b * alpha_dot) / I
    alpha_dot_next = alpha_dot + alpha_ddot * dt
    alpha_next = alpha + alpha_dot_next * dt
    return alpha_next, alpha_dot_next

def alpha_from_kappa(v, kappa, g, alpha_max_rad):
    ay = v * v * kappa
    alpha_d = math.atan2(ay, g)
    alpha_d = clamp(alpha_d, -alpha_max_rad, alpha_max_rad)
    return alpha_d, ay

def alpha_from_yaw_rate(v, yaw_rate, g, alpha_max_rad):
    ay = v * yaw_rate
    alpha_d = math.atan2(ay, g)
    alpha_d = clamp(alpha_d, -alpha_max_rad, alpha_max_rad)
    return alpha_d, ay

def yaw_rate_measurement(v, kappa_true, rng, bias=0.0, noise_std=0.0):
    r_true = v * kappa_true
    noise = rng.normal(0.0, noise_std) if noise_std > 0.0 else 0.0
    r_meas = r_true + bias + noise
    return r_true, r_meas

def kappa_est_from_yaw(v, yaw_rate):
    if abs(v) < V_EPS:
        return 0.0
    return yaw_rate / v

def ensure_clean_plots_dir():
    if not SAVE_PNG:
        return
    os.makedirs(PLOTS_DIR, exist_ok=True)

def clear_old_pngs():
    if not SAVE_PNG:
        return
    if not os.path.isdir(PLOTS_DIR):
        return
    for fn in os.listdir(PLOTS_DIR):
        if fn.lower().endswith(".png"):
            try:
                os.remove(os.path.join(PLOTS_DIR, fn))
            except OSError:
                pass

def save_current_figure(filename):
    if not SAVE_PNG:
        return
    ensure_clean_plots_dir()
    out_path = os.path.join(PLOTS_DIR, filename)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")

# ============================================================
# SCENARIOS
# ============================================================

def scenario_straight(t):
    return 15.0, 0.0

def scenario_constant_turn(t):
    return 15.0, 1.0 / 60.0

def scenario_s_curve(t):
    return 15.0, 0.02 * math.sin(0.5 * t)

def scenario_saturation(t):
    return 25.0, 1.0 / 25.0

# ============================================================
# SIMULATION
# ============================================================

def run_sim(scenario_fn, name="", T=T_SIM, dt=DT, seed=RNG_SEED):
    p = LeanControllerParams()
    rng = np.random.default_rng(seed)

    n = int(T / dt) + 1
    t_vec = np.linspace(0.0, T, n)

    v_k = v_r = 0.0
    alpha_k = alpha_r = 0.0
    alpha_dot_k = alpha_dot_r = 0.0
    yaw_filt = 0.0

    log = {
        "t": np.zeros(n),
        "v_des": np.zeros(n),
        "kappa_true": np.zeros(n),
        "kappa_est": np.zeros(n),
        "yaw_rate_filt": np.zeros(n),
        "v_k": np.zeros(n),
        "v_r": np.zeros(n),
        "alpha_k": np.zeros(n),
        "alpha_r": np.zeros(n),
        "alpha_d_k": np.zeros(n),
        "alpha_d_r": np.zeros(n),
        "fy_k": np.zeros(n),
        "fy_r": np.zeros(n),
    }

    for i, ti in enumerate(t_vec):
        v_des, kappa_true = scenario_fn(ti)

        # MAP PATH
        v_max_k = speed_limit_for_curvature(kappa_true, p.alpha_max_rad, p.g)
        v_k = speed_dynamics_step(v_k, min(v_des, v_max_k), p.v_track_tau, dt)

        alpha_d_k, ay_k = alpha_from_kappa(v_k, kappa_true, p.g, p.alpha_max_rad)
        tau_k = p.Kp * (alpha_d_k - alpha_k) - p.Kd * alpha_dot_k
        alpha_k, alpha_dot_k = roll_dynamics_step(alpha_k, alpha_dot_k, tau_k, p.I, p.b, dt)

        fy_k = ay_k * math.cos(alpha_k) - p.g * math.sin(alpha_k)

        # YAW-ONLY PATH
        kappa_prev = log["kappa_est"][i - 1] if i > 0 else 0.0
        v_max_r = speed_limit_for_curvature(kappa_prev, p.alpha_max_rad, p.g)
        v_r = speed_dynamics_step(v_r, min(v_des, v_max_r), p.v_track_tau, dt)

        _, yaw_meas = yaw_rate_measurement(
            v_r, kappa_true, rng, YAW_BIAS_RAD_S, YAW_NOISE_STD
        )
        yaw_filt = lowpass_1st_order(yaw_filt, yaw_meas, YAW_FILTER_TAU_S, dt)
        kappa_est = kappa_est_from_yaw(v_r, yaw_filt)

        alpha_d_r, ay_r = alpha_from_yaw_rate(v_r, yaw_filt, p.g, p.alpha_max_rad)
        tau_r = p.Kp * (alpha_d_r - alpha_r) - p.Kd * alpha_dot_r
        alpha_r, alpha_dot_r = roll_dynamics_step(alpha_r, alpha_dot_r, tau_r, p.I, p.b, dt)

        fy_r = ay_r * math.cos(alpha_r) - p.g * math.sin(alpha_r)

        # LOG
        log["t"][i] = ti
        log["v_des"][i] = v_des
        log["kappa_true"][i] = kappa_true
        log["kappa_est"][i] = kappa_est
        log["yaw_rate_filt"][i] = yaw_filt
        log["v_k"][i] = v_k
        log["v_r"][i] = v_r
        log["alpha_k"][i] = alpha_k
        log["alpha_r"][i] = alpha_r
        log["alpha_d_k"][i] = alpha_d_k
        log["alpha_d_r"][i] = alpha_d_r
        log["fy_k"][i] = fy_k
        log["fy_r"][i] = fy_r

    return log, p

# ============================================================
# PLOTS (4 PER SCENARIO)
# ============================================================

def plot_log_4(log, p, name, fig_base):
    t = log["t"]

    plt.figure(fig_base + 0)
    plt.plot(t, log["kappa_true"], label="kappa_true")
    plt.plot(t, log["kappa_est"], label="kappa_est (yaw)", linestyle=":")
    plt.legend(); plt.grid(); plt.title(f"{name}: curvature")
    save_current_figure(f"{name}_1_curvature.png")

    plt.figure(fig_base + 1)
    plt.plot(t, log["v_k"], label="v_map")
    plt.plot(t, log["v_r"], label="v_yaw")
    plt.legend(); plt.grid(); plt.title(f"{name}: speed")
    save_current_figure(f"{name}_2_speed_limit.png")

    plt.figure(fig_base + 2)
    plt.plot(t, log["fy_k"], label="f_y seat (map)")
    plt.plot(t, log["fy_r"], label="f_y seat (yaw)", linestyle=":")
    plt.axhline(0.0, linestyle="--", color="k")
    plt.legend(); plt.grid(); plt.title(f"{name}: comfort metric")
    save_current_figure(f"{name}_3_lateral_accel.png")

    plt.figure(fig_base + 3)
    plt.plot(t, np.degrees(log["alpha_k"]), label="alpha_map")
    plt.plot(t, np.degrees(log["alpha_d_k"]), linestyle=":", label="alpha_d_map")
    plt.plot(t, np.degrees(log["alpha_r"]), label="alpha_yaw")
    plt.plot(t, np.degrees(log["alpha_d_r"]), linestyle=":", label="alpha_d_yaw")
    plt.axhline(p.alpha_max_deg, linestyle="--", color="k")
    plt.axhline(-p.alpha_max_deg, linestyle="--", color="k")
    plt.legend(); plt.grid(); plt.title(f"{name}: seat roll")
    save_current_figure(f"{name}_4_lean_tracking.png")

# ============================================================
# MAIN
# ============================================================

def main():
    print("RUNNING:", os.path.abspath(__file__))
    print("CWD:", os.getcwd())

    if SAVE_PNG:
        ensure_clean_plots_dir()
        clear_old_pngs()

    plt.close("all")

    tests = [
        ("straight", scenario_straight),
        ("constant_turn", scenario_constant_turn),
        ("s_curve", scenario_s_curve),
        ("saturation", scenario_saturation),
    ]

    for i, (name, fn) in enumerate(tests):
        log, p = run_sim(fn, name)
        print(
            f"{name}: max |f_y| map = {np.max(np.abs(log['fy_k'])):.3f}, "
            f"yaw = {np.max(np.abs(log['fy_r'])):.3f}"
        )
        plot_log_4(log, p, name, fig_base=100 + 10 * i)

    if SHOW_WINDOWS:
        plt.show()
    else:
        plt.close("all")

    if SAVE_PNG:
        print(f"Saved plots to .\\{PLOTS_DIR}\\")

if __name__ == "__main__":
    main()
