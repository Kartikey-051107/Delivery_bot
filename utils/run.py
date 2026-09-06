#!/usr/bin/env python3
"""
run.py  --  Solve every level with YOUR dp.py and draw all the figures.

By the Electronics & Robotics Club, BITS Goa.

Run it from the assignment/ folder once check.py is happy:

    conda activate rl
    python run.py

It calls your functions in dp.py, prints a summary of how each algorithm did, and
saves every figure into the figures/ folder (headless, so it also works on WSL):

    level1_value_policy.png     your optimal value map and policy for Level 1
    pi_vs_vi.png                the policy-iteration vs value-iteration race
    level2_value_policy.png     the slippery-floor solution
    slip_panel.png              how the route changes as the floor gets slippery
    level3_value_policy.png     the pickup-then-deliver plan (two panels)
    rollout.mp4                 your robot running its optimal policy (video)
    async_policy.png            in-place vs two-array: who finds the policy first
    modified_pi.png             the k dial from value iteration to policy iteration
    curse.png                   solve time as the state space grows

Upload your Level 3 rollout.mp4 to Google Classroom when you hand in.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

# This file lives in the provided "utils" folder. Let it import the provided
# modules (siblings here) and your dp.py (in the project root, one level up).
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent))

import numpy as np

from env import GridWorld
import maps
import plotting
import dp


def build(name, **overrides):
    cfg = dict(maps.CONFIGS[name])
    cfg.pop("title")
    cfg.update(overrides)
    gamma = cfg.pop("gamma")
    return GridWorld(gamma=gamma, **cfg), gamma


def q(env, V, s, a, g):
    return sum(p * (r + g * V[s2]) for p, s2, r in env.transitions(s, a))


# --- small instrumented solvers, only used to draw the convergence curves ----
def precise_vstar(env, g, theta=1e-8, max_sweeps=5000):
    """A high-accuracy optimal value function, used only as the error baseline
    for the convergence plots so every method's floor is the same true optimum."""
    V = np.zeros(env.n_states)
    for _ in range(max_sweeps):
        Vn = V.copy(); delta = 0.0
        for s in env.states:
            if env.is_terminal(s):
                continue
            Vn[s] = max(q(env, V, s, a, g) for a in env.actions)
            delta = max(delta, abs(Vn[s] - V[s]))
        V = Vn
        if delta < theta:
            break
    return V


def vi_error_curve(env, g, Vstar, max_sweeps=120):
    V = np.zeros(env.n_states)
    errs = []
    for _ in range(max_sweeps):
        errs.append(float(np.abs(V - Vstar).max()))
        Vn = V.copy()
        for s in env.states:
            if env.is_terminal(s):
                continue
            Vn[s] = max(q(env, V, s, a, g) for a in env.actions)
        V = Vn
        if errs[-1] < 1e-6:
            break
    return np.array(errs)


def pi_error_curve(env, g, Vstar, theta=1e-6):
    policy = {s: "up" for s in env.states if not env.is_terminal(s)}
    xs, ys, total = [], [], 0
    for _ in range(40):
        V = np.zeros(env.n_states)
        while True:
            Vn = V.copy(); delta = 0.0
            for s in env.states:
                if env.is_terminal(s):
                    continue
                Vn[s] = q(env, V, s, policy[s], g)
                delta = max(delta, abs(Vn[s] - V[s]))
            V = Vn; total += 1
            if delta < theta:
                break
        xs.append(total); ys.append(float(np.abs(V - Vstar).max()))
        new = {s: max(env.actions, key=lambda a: q(env, V, s, a, g))
               for s in env.states if not env.is_terminal(s)}
        if new == policy:
            break
        policy = new
    return xs, np.maximum(ys, 1e-12)


def mpi_curve(env, g, Vstar, k, theta=1e-6, max_iters=400):
    V = np.zeros(env.n_states)
    xs, ys, total = [], [], 0
    for _ in range(max_iters):
        policy = {s: max(env.actions, key=lambda a: q(env, V, s, a, g))
                  for s in env.states if not env.is_terminal(s)}
        for _ in range(k):
            Vn = V.copy(); delta = 0.0
            for s in env.states:
                if env.is_terminal(s):
                    continue
                Vn[s] = q(env, V, s, policy[s], g)
                delta = max(delta, abs(Vn[s] - V[s]))
            V = Vn; total += 1
            if delta < theta:
                break
        xs.append(total); ys.append(float(np.abs(V - Vstar).max()))
        if ys[-1] < 1e-6:
            break
    return xs, np.maximum(ys, 1e-12)


def async_wrong_curve(env, g, opt_actions, order, inplace, max_sweeps=120):
    V = np.zeros(env.n_states)
    states = [s for s in order if not env.is_terminal(s)]
    wrong = []
    for _ in range(max_sweeps):
        if inplace:
            for s in states:
                V[s] = max(q(env, V, s, a, g) for a in env.actions)
        else:
            Vn = V.copy()
            for s in states:
                Vn[s] = max(q(env, V, s, a, g) for a in env.actions)
            V = Vn
        n_wrong = sum(
            1 for s in states
            if max(env.actions, key=lambda a: q(env, V, s, a, g)) != opt_actions[s]
        )
        wrong.append(n_wrong)
        if n_wrong == 0:
            break
    return wrong


def optimal_actions(env, g, Vstar):
    return {s: max(env.actions, key=lambda a: q(env, Vstar, s, a, g))
            for s in env.states if not env.is_terminal(s)}


def main():
    print("Solving every level with your dp.py ...\n")
    summary = []

    # ---------------- Level 1: deterministic --------------------------------
    env, g = build("level1")
    V, pol, sweeps = dp.value_iteration(env, g)
    _, _, pi_iters = dp.policy_iteration(env, g)
    plotting.figure_value_policy(env, V, pol, "level1_value_policy.png",
                                 maps.CONFIGS["level1"]["title"])
    Vstar = precise_vstar(env, g)  # accurate baseline for the convergence curves
    vi_err = vi_error_curve(env, g, Vstar)
    pix, piy = pi_error_curve(env, g, Vstar)
    plotting.figure_pi_vs_vi(vi_err, pix, piy)
    summary.append(("level1", env.n_states, f"VI {sweeps} sweeps", f"PI {pi_iters} iters"))

    # ---------------- Level 2: slippery -------------------------------------
    env2, g2 = build("level2")
    V2, pol2, sweeps2 = dp.value_iteration(env2, g2)
    plotting.figure_value_policy(env2, V2, pol2, "level2_value_policy.png",
                                 maps.CONFIGS["level2"]["title"])
    panel = []
    for p in maps.SLIP_SWEEP:
        ep, gp = build("level2", slip=p)
        Vp, polp, _ = dp.value_iteration(ep, gp)
        panel.append((ep, Vp, polp, f"slip p = {p:.2f}"))
    plotting.figure_slip_panel(panel)
    summary.append(("level2", env2.n_states, f"VI {sweeps2} sweeps",
                    f"V*(start) {V2[env2.start_state]:.1f}"))

    # ---------------- Level 3: pickup then deliver --------------------------
    env3, g3 = build("level3")
    V3, pol3, sweeps3 = dp.value_iteration(env3, g3)
    plotting.figure_value_policy(env3, V3, pol3, "level3_value_policy.png",
                                 maps.CONFIGS["level3"]["title"])
    plotting.figure_rollout(env3, V3, pol3, "rollout.mp4", seed=2)

    # asynchronous DP: two-array vs in-place vs in-place goal-first ordering.
    # Shown on the Level 2 maze (big enough to separate the curves, quick to sweep).
    V2p = precise_vstar(env2, g2)
    opt = optimal_actions(env2, g2, V2p)
    fwd = list(env2.states)
    goal_first = list(np.argsort(-V2p))  # highest value (nearest goal) first
    curves = [
        ("two-array (synchronous)", "slate",
         async_wrong_curve(env2, g2, opt, fwd, inplace=False)),
        ("in-place, natural order", "blue",
         async_wrong_curve(env2, g2, opt, fwd, inplace=True)),
        ("in-place, goal-first order", "orange",
         async_wrong_curve(env2, g2, opt, goal_first, inplace=True)),
    ]
    plotting.figure_async_policy(curves)

    # modified policy iteration: the k dial (on the small map, it is quick)
    mcurves = []
    for k, col, lab in [(1, "blue", r"$k=1$ (value iteration)"),
                        (3, "teal", r"$k=3$"),
                        (10, "purple", r"$k=10$"),
                        (10 ** 9, "orange", r"$k=\infty$ (policy iteration)")]:
        xs, ys = mpi_curve(env, g, Vstar, min(k, 10 ** 6))
        mcurves.append((lab, col, xs, ys))
    plotting.figure_modified_pi(mcurves)
    summary.append(("level3", env3.n_states, f"VI {sweeps3} sweeps",
                    f"V*(start) {V3[env3.start_state]:.1f}"))

    # ---------------- Curse of dimensionality -------------------------------
    counts, times, labels = [], [], []
    for name, pkg in [("level1", False), ("level2", False),
                      ("grand", False), ("grand", True)]:
        e, gg = build(name, has_package_dim=pkg)
        t0 = time.time()
        dp.value_iteration(e, gg)
        dt = time.time() - t0
        counts.append(e.n_states); times.append(dt)
        labels.append(f"{name}{' +pkg' if pkg else ''}")
    order = np.argsort(counts)
    plotting.figure_curse(list(np.array(counts)[order]), list(np.array(times)[order]),
                          [labels[i] for i in order])

    # ---------------- Summary ----------------------------------------------
    print("\nSummary")
    print("-------")
    for name, n, a, b in summary:
        print(f"  {name:12s} states={n:5d}   {a:16s} {b}")
    figdir = plotting.FIGDIR
    print(f"\nAll figures written to: {figdir}")
    print(f"Watch your robot solve the maze:  {figdir / 'rollout.mp4'}")
    print("  Ubuntu:  xdg-open figures/rollout.mp4")
    print("  WSL:     explorer.exe figures   (then double-click rollout.mp4)")


if __name__ == "__main__":
    main()
