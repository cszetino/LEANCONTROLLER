"""
LEAN-INTO-CORNER CONTROLLER + DEMO HARNESS (MINIMAL PLOTS + OCCUPANT "DOOR-SLAM" METRIC)

Includes:
- Controller with curvature estimate kappa_hat ~= yaw_rate_r / v
- Desired lean alpha_d = atan(a_y/g), with alpha_max + alpha_rate_max constraints
- Fallback speed cap v_max = sqrt( g*tan(alpha_max) / |kappa| ), suppressed near straight driving
- Simple roll plant + PD tracking torque for simulation
- Minimal plots + direct proof signal:
    a_side_seat = a_y*cos(alpha) - g*sin(alpha)   (target ~ 0)

Run:
  python compiled_lean_demo_minplots.py

Tip for profiling:
  set PLOT = False before running cProfile
"""

import math
import time
from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt


# Toggle plotting for speed/profiling
PLOT = True


# -------------------------
# Utilities
# -------------------------
def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


# -------------------------
# Controller (Part A + optional torque for plant tracking)
# -------------------------
@dataclass
class LeanControllerParams:
    g: float = 9.81

    # comfort + actuator constraints
    alpha_max_rad: float = math.radians(10.0)     # max tilt angle
    alpha_rate_max: float = math.radians(30.0)    # max tilt rate (rad/s)
    tau_max: float = 800.0                        # torque limit (demo)

    # simple roll plant (seat/cabin actuator model)
    I: float = 120.0
    b: float = 40.0

    # tracking gains
    kp: float = 8.0
    kd: float = 2.0

    # numerical safety
    v_min_for_kappa: float = 0.5   # avoid r/v blow-up near zero speed (m/s)
    kappa_max: float = 0.5         # clamp curvature estimate (1/m)

    # speed-cap suppression near straight driving
    kappa_min_for_cap: float = 1e-3   # below this, treat as "no cap" (1/m)
    cap_value_when_straight: float = float("nan")  # NaN makes plots skip those points


class LeanIntoCornerController:
    """
    Part (a) core:
      - estimate curvature kappa_hat ~ yaw_rate_r / v
      - compute desired lean alpha_d = atan( (v^2*kappa_hat)/g )
      - apply alpha_max and alpha_rate_max constraints
      - compute fallback speed cap v_max (planner/braking can enforce)

    Optional for simulation:
      - PD tracking to generate torque tau for roll plant: I*alpha_ddot + b*alpha_dot = tau
    """

    def __init__(self, params: LeanControllerParams):
        self.p = params
        self._alpha_d_prev = 0.0

    def estimate_curvature(self, v: float, yaw_rate_r: float) -> float:
        if abs(v) < self.p.v_min_for_kappa:
            return 0.0
        kappa = yaw_rate_r / v
        return clamp(kappa, -self.p.kappa_max, self.p.kappa_max)

    def fallback_speed_cap(self, kappa_hat: float) -> float:
        # Suppress cap near straight driving to avoid huge spikes when |kappa| -> 0
        if abs(kappa_hat) < self.p.kappa_min_for_cap:
            return self.p.cap_value_when_straight

        a_y_max = self.p.g * math.tan(self.p.alpha_max_rad)
        return math.sqrt(a_y_max / abs(kappa_hat))

    def step(self, v: float, yaw_rate_r: float, alpha: float, alpha_dot: float, dt: float):
        if dt <= 0.0:
            raise ValueError("dt must be positive")

        # 1) curvature estimate
        kappa_hat = self.estimate_curvature(v, yaw_rate_r)

        # 2) lateral accel from lecture: a_y = v^2 * kappa
        a_y = (v * v) * kappa_hat

        # 3) desired lean: alpha_d = atan(a_y/g)
        alpha_d_raw = math.atan2(a_y, self.p.g)

        # 4) constraint: max tilt angle
        alpha_d_cmd = clamp(alpha_d_raw, -self.p.alpha_max_rad, self.p.alpha_max_rad)

        # 5) constraint: max tilt rate
        prev = self._alpha_d_prev
        max_step = self.p.alpha_rate_max * dt
        alpha_d = clamp(alpha_d_cmd, prev - max_step, prev + max_step)
        alpha_d_dot = (alpha_d - prev) / dt
        self._alpha_d_prev = alpha_d

        # 6) PD tracking -> torque for roll plant
        tau = (
            self.p.I * (self.p.kp * (alpha_d - alpha) + self.p.kd * (alpha_d_dot - alpha_dot))
            + self.p.b * alpha_dot
        )
        tau = clamp(tau, -self.p.tau_max, self.p.tau_max)

        # 7) speed cap enhancement
        v_max = self.fallback_speed_cap(kappa_hat)

        return tau, alpha_d, kappa_hat, v_max, a_y


# -------------------------
# Simple Roll Plant for Simulation
# -------------------------
def roll_dynamics_step(alpha: float, alpha_dot: float, tau: float, I: float, b: float, dt: float):
    alpha_ddot = (tau - b * alpha_dot) / I
    alpha_dot_next = alpha_dot + alpha_ddot * dt
    alpha_next = alpha + alpha_dot_next * dt
    return alpha_next, alpha_dot_next


# -------------------------
# Scenarios (generate v, yaw_rate_r)
# -------------------------
def scenario_straight(t: float):
    v = 15.0
    r = 0.0
    return v, r

def scenario_constant_turn(t: float):
    v = 15.0
    rho = 60.0
    r = v / rho
    return v, r

def scenario_s_curve(t: float):
    v = 15.0
    r = 0.25 * math.sin(0.5 * t)
    return v, r

def scenario_tight_fast(t: float):
    v = 25.0
    rho = 25.0
    r = v / rho
    return v, r


# -------------------------
# Simulation Runner
# -------------------------
def run_sim(scenario_fn, T: float = 20.0, dt: float = 0.01):
    p = LeanControllerParams()
    ctrl = LeanIntoCornerController(p)

    n = int(T / dt)
    t_arr = np.linspace(0.0, T, n, endpoint=False)

    alpha = 0.0
    alpha_dot = 0.0

    # Minimal logging (only what we need)
    log = {k: [] for k in [
        "t", "v", "kappa_hat", "alpha", "alpha_dot", "alpha_d", "v_max",
        "a_y", "a_side_seat"
    ]}

    for t in t_arr:
        v, yaw_rate_r = scenario_fn(t)

        tau, alpha_d, kappa_hat, v_max, a_y = ctrl.step(
            v=v, yaw_rate_r=yaw_rate_r, alpha=alpha, alpha_dot=alpha_dot, dt=dt
        )

        # Update roll plant (actual seat/cabin tilt)
        alpha, alpha_dot = roll_dynamics_step(alpha, alpha_dot, tau, p.I, p.b, dt)

        # Occupant "door-slam" metric: seat-frame lateral acceleration
        # target: ~0
        a_side_seat = a_y * math.cos(alpha) - p.g * math.sin(alpha)

        # Log
        log["t"].append(t)
        log["v"].append(v)
        log["kappa_hat"].append(kappa_hat)
        log["alpha"].append(alpha)
        log["alpha_dot"].append(alpha_dot)
        log["alpha_d"].append(alpha_d)
        log["v_max"].append(v_max)
        log["a_y"].append(a_y)
        log["a_side_seat"].append(a_side_seat)

    for k in log:
        log[k] = np.array(log[k], dtype=float)

    # Quick numeric proof (useful even if plots are off)
    a = log["a_side_seat"]
    metrics = {
        "a_side_seat_rms": float(np.sqrt(np.mean(a**2))),
        "a_side_seat_max_abs": float(np.max(np.abs(a))),
    }

    return log, p, metrics


# -------------------------
# Minimal Plotting
# -------------------------
def plot_log_minimal(log, title: str):
    t = log["t"]

    # 1) Curvature estimate
    plt.figure()
    plt.plot(t, log["kappa_hat"])
    plt.xlabel("time (s)")
    plt.ylabel("curvature kappa_hat (1/m)")
    plt.title(f"{title}: curvature estimate")

    # 2) Lean tracking
    plt.figure()
    plt.plot(t, np.degrees(log["alpha"]), label="alpha (actual)")
    plt.plot(t, np.degrees(log["alpha_d"]), label="alpha_d (command)", linestyle=":")
    plt.xlabel("time (s)")
    plt.ylabel("lean angle (deg)")
    plt.title(f"{title}: lean tracking")
    plt.legend()

    # 3) Objective proof: occupant sideways push
    plt.figure()
    plt.plot(t, log["a_side_seat"])
    plt.xlabel("time (s)")
    plt.ylabel("a_side_seat (m/s^2)")
    plt.title(f"{title}: occupant sideways push (target ~ 0)")

    # 4) Enhancement: speed vs fallback cap (NaNs will be skipped)
    plt.figure()
    plt.plot(t, log["v"], label="v")
    plt.plot(t, log["v_max"], label="v_max (fallback cap)", linestyle=":")
    plt.xlabel("time (s)")
    plt.ylabel("speed (m/s)")
    plt.title(f"{title}: speed vs fallback cap")
    plt.legend()


# -------------------------
# Main
# -------------------------
def main():
    tests = [
        ("straight", scenario_straight),
        ("constant_turn", scenario_constant_turn),
        ("s_curve", scenario_s_curve),
        ("tight_fast", scenario_tight_fast),
    ]

    for name, fn in tests:
        t0 = time.perf_counter()
        log, _, metrics = run_sim(fn, T=20.0, dt=0.01)
        t1 = time.perf_counter()

        print(f"{name}: runtime {t1 - t0:.4f} s for {len(log['t'])} steps")
        print(f"  a_side_seat RMS: {metrics['a_side_seat_rms']:.4f} m/s^2")
        print(f"  max |a_side_seat|: {metrics['a_side_seat_max_abs']:.4f} m/s^2")

        if PLOT:
            plot_log_minimal(log, name)

    if PLOT:
        plt.show()


if __name__ == "__main__":
    main()
