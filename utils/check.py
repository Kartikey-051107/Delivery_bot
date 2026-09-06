#!/usr/bin/env python3
"""
check.py  --  Grade your dp.py against the trusted answers.

By the Electronics & Robotics Club, BITS Goa.

Run it from inside the activated environment, in the assignment/ folder:

    conda activate rl
    python check.py

It runs each of your six functions on the graded maps and compares the results to
stored reference answers, and to an independent Monte Carlo simulation. Every line
tells you exactly which function passed or failed. When everything passes you are
done. A function you have not written yet shows up as "not implemented yet", not a
crash, so you can fill them in one at a time and re-run.
"""

from __future__ import annotations

import sys
from pathlib import Path

# This file lives in the provided "utils" folder. Let it import the provided
# modules (siblings here) and your dp.py (in the project root, one level up).
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import numpy as np

from env import GridWorld, uniform_random_policy
import maps
import dp
PASS, FAIL, SKIP = "[ OK ]", "[FAIL]", "[----]"
VALUE_TOL = 1e-2   # values match the reference this closely
Q_TOL = 1e-4       # a greedy action must be this close to the best q value


def build_env(name):
    cfg = dict(maps.CONFIGS[name])
    cfg.pop("title")
    gamma = cfg.pop("gamma")
    return GridWorld(gamma=gamma, **cfg), gamma


def q(env, V, s, a, gamma):
    return sum(p * (r + gamma * V[s2]) for p, s2, r in env.transitions(s, a))


def policy_is_optimal(env, policy, Vstar, gamma):
    """Every chosen action is within Q_TOL of the best possible q value at V*."""
    for s in env.states:
        if env.is_terminal(s):
            continue
        a = max(policy[s], key=policy[s].get)
        best = max(q(env, Vstar, s, aa, gamma) for aa in env.actions)
        if q(env, Vstar, s, a, gamma) < best - Q_TOL:
            return False
    return True


def section(title):
    print("\n" + title)
    print("-" * len(title))


class Results:
    def __init__(self):
        self.total = 0
        self.passed = 0

    def record(self, ok, label, detail=""):
        self.total += 1
        self.passed += int(ok)
        mark = PASS if ok else FAIL
        print(f"  {mark} {label}" + (f"   ({detail})" if detail else ""))


def run(label, fn, results):
    """Run one graded check. A missing function fails softly, not with a crash."""
    try:
        ok, detail = fn()
    except NotImplementedError:
        results.total += 1
        print(f"  {SKIP} {label}   (not implemented yet)")
        return
    except Exception as exc:  # any other bug in the student code
        results.record(False, label, f"error: {exc}")
        return
    results.record(ok, label, detail)


def main():
    print("Delivery Robot: grading your dp.py")
    print("By the Electronics & Robotics Club, BITS Goa.")

    ref_path = HERE / "reference_answers.npz"
    if not ref_path.exists():
        print("\nCould not find reference_answers.npz. Ask a mentor to rebuild it.")
        return 1
    ref = np.load(ref_path)

    r = Results()

    # ---- Task 1: policy evaluation (prediction) ------------------------------
    section("Task 1  policy_evaluation  (prediction)")
    env1, g1 = build_env("level1")

    def t_eval():
        V = dp.policy_evaluation(env1, uniform_random_policy(env1), g1)
        exp = ref["level1_Vuniform"]
        err = float(np.abs(V - exp).max())
        return err < VALUE_TOL, f"max error to reference = {err:.2e}"
    run("evaluate the uniform-random policy on level1", t_eval, r)

    # ---- Task 2: greedy policy (improvement) ---------------------------------
    section("Task 2  greedy_policy  (improvement step)")

    def t_greedy():
        Vstar = ref["level1_Vstar"]
        pol = dp.greedy_policy(env1, Vstar, g1)
        return policy_is_optimal(env1, pol, Vstar, g1), "greedy w.r.t. V* is optimal"
    run("greedy_policy picks the best action at V* (level1)", t_greedy, r)

    # ---- Task 3: value iteration ---------------------------------------------
    section("Task 3  value_iteration  (control)")
    for name in ("level1", "level2", "level3"):
        env, g = build_env(name)
        Vstar = ref[f"{name}_Vstar"]

        def t_vi(env=env, g=g, Vstar=Vstar):
            V, pol, sweeps = dp.value_iteration(env, g)
            err = float(np.abs(V - Vstar).max())
            ok = err < VALUE_TOL and policy_is_optimal(env, pol, Vstar, g) and sweeps > 0
            return ok, f"max error = {err:.2e}, sweeps = {sweeps}"
        run(f"value_iteration solves {name}", t_vi, r)

    # ---- Task 4: policy iteration --------------------------------------------
    section("Task 4  policy_iteration  (control)")
    for name in ("level1", "level2"):
        env, g = build_env(name)
        Vstar = ref[f"{name}_Vstar"]

        def t_pi(env=env, g=g, Vstar=Vstar):
            V, pol, iters = dp.policy_iteration(env, g)
            err = float(np.abs(V - Vstar).max())
            ok = err < VALUE_TOL and policy_is_optimal(env, pol, Vstar, g) and iters > 0
            return ok, f"max error = {err:.2e}, iterations = {iters}"
        run(f"policy_iteration solves {name}", t_pi, r)

    # ---- Task 5: in-place (asynchronous) value iteration ---------------------
    section("Task 5  value_iteration_inplace  (asynchronous DP)")
    for name in ("level1", "level2"):
        env, g = build_env(name)
        Vstar = ref[f"{name}_Vstar"]

        def t_ip(env=env, g=g, Vstar=Vstar):
            V, pol, sweeps = dp.value_iteration_inplace(env, g)
            err = float(np.abs(V - Vstar).max())
            return err < VALUE_TOL, f"max error = {err:.2e}, sweeps = {sweeps}"
        run(f"value_iteration_inplace solves {name}", t_ip, r)

    # ---- Task 6: modified policy iteration -----------------------------------
    section("Task 6  modified_policy_iteration  (the k dial)")
    Vstar1 = ref["level1_Vstar"]
    for k in (1, 5, 50):
        def t_mpi(k=k):
            V, pol, iters = dp.modified_policy_iteration(env1, g1, k)
            err = float(np.abs(V - Vstar1).max())
            return err < VALUE_TOL, f"max error = {err:.2e}, iterations = {iters}"
        run(f"modified_policy_iteration (k={k}) solves level1", t_mpi, r)

    # ---- Cross-checks that need no reference ---------------------------------
    section("Cross-checks  (these need no stored answers)")

    def t_agree():
        Vv, _, _ = dp.value_iteration(env1, g1)
        Vp, _, _ = dp.policy_iteration(env1, g1)
        Vi, _, _ = dp.value_iteration_inplace(env1, g1)
        d = max(float(np.abs(Vv - Vp).max()), float(np.abs(Vv - Vi).max()))
        return d < VALUE_TOL, f"VI vs PI vs in-place differ by {d:.2e}"
    run("value iteration, policy iteration and in-place agree", t_agree, r)

    def t_mc():
        # Week 0 idea: the value at the start equals the average return you get by
        # actually driving the optimal policy many times.
        env2, g2 = build_env("level2")
        Vstar2 = ref["level2_Vstar"]
        _, pol, _ = dp.value_iteration(env2, g2)
        rng = np.random.default_rng(0)
        N, returns = 4000, []
        for i in range(N):
            s, G_ret, disc = env2.start_state, 0.0, 1.0
            for _ in range(600):
                if env2.is_terminal(s):
                    break
                a = max(pol[s], key=pol[s].get)
                outs = env2.transitions(s, a)
                j = rng.choice(len(outs), p=[p for p, _, _ in outs])
                _, s, rew = outs[j]
                G_ret += disc * rew
                disc *= g2
            returns.append(G_ret)
        mean = float(np.mean(returns))
        stderr = float(np.std(returns) / np.sqrt(N))
        target = float(Vstar2[env2.start_state])
        ok = abs(mean - target) < 0.6 + 4 * stderr
        return ok, f"MC mean = {mean:.2f} vs V*(start) = {target:.2f} (+-{stderr:.2f})"
    run("Monte Carlo return of the optimal policy matches V*(start)", t_mc, r)

    # ---- Summary -------------------------------------------------------------
    section("Result")
    print(f"  Passed {r.passed} of {r.total} checks.")
    if r.passed == r.total:
        print("  Every check passed. Your Dynamic Programming solver is correct.")
        print("  Now run  python run.py  to make your figures.")
        return 0
    print("  Some checks did not pass yet. Fix the functions marked [FAIL] or")
    print("  [----] above and run this file again. Stuck for a while? That is normal.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
