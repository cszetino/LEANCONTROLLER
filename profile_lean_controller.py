import importlib
import cProfile
import pstats
import io
import time


TARGET_MODULE = "script"
T = 20.0
DT = 0.01
TOP_N = 30


def load_target():
    m = importlib.import_module(TARGET_MODULE)
    required = [
        "run_sim",
        "scenario_straight",
        "scenario_constant_turn",
        "scenario_s_curve",
        "scenario_saturation",
    ]
    missing = [x for x in required if not hasattr(m, x)]
    if missing:
        raise RuntimeError(f"script.py missing: {missing}")
    return m


def profile_call(fn, sort_by="cumtime", top_n=TOP_N):
    pr = cProfile.Profile()
    pr.enable()
    fn()
    pr.disable()

    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).strip_dirs().sort_stats(sort_by)
    ps.print_stats(top_n)
    return s.getvalue()


def bench_end_to_end(m):
    n_steps = int(T / DT) + 1
    t0 = time.perf_counter()
    m.run_sim(m.scenario_s_curve, name="bench", T=T, dt=DT, seed=1)
    t1 = time.perf_counter()

    total = t1 - t0
    per_step = total / max(1, n_steps)
    hz = 1.0 / per_step if per_step > 0 else float("inf")

    print("===================================================")
    print("Headline timing (end-to-end)")
    print("===================================================")
    print(f"Sim runtime: {total:.4f} s for {n_steps} steps")
    print(f"Time/step : {per_step * 1e6:.2f} us/step")
    print(f"Rate      : {hz:.0f} Hz")
    print()


def main():
    m = load_target()
    bench_end_to_end(m)

    scenarios = [
        ("straight", m.scenario_straight),
        ("constant_turn", m.scenario_constant_turn),
        ("s_curve", m.scenario_s_curve),
        ("saturation", m.scenario_saturation),
    ]

    print("===================================================")
    print("cProfile hotspots (by cumulative time)")
    print("===================================================")

    for name, fn in scenarios:
        print(f"\n--- Scenario: {name} ---")
        print(profile_call(lambda: m.run_sim(fn, name=name, T=T, dt=DT, seed=1)))


if __name__ == "__main__":
    main()
