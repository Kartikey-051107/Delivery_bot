"""
dp.py  --  YOUR Dynamic Programming solver for the Delivery Robot.

======================================================================
   THIS IS THE ONE FILE YOU EDIT. Everything else is plumbing.
======================================================================

You will write six functions. Each one turns an idea from the Week 2 notes into a
loop over states. Work out the loop yourself from the notes; the docstring here only
tells you what the function must return and points you to the right chapter.

The world is handed to you as `env` (see env.py). The three things you need:

    env.states               a list of integer state ids, 0 .. n_states - 1
    env.actions              ["up", "down", "left", "right"]
    env.is_terminal(s)       True once the robot has delivered or crashed
    env.transitions(s, a)    a list of (probability, next_state, reward) triples

A value function V is a 1-D NumPy array: V[s] is the value of state s. Terminal
states always have value 0.

A policy is a dictionary mapping each non-terminal state to a distribution over
actions, for example {"up": 0.25, "down": 0.25, "left": 0.25, "right": 0.25}. A
deterministic policy just puts probability 1.0 on one action. The helpers
env.deterministic(...) and env.uniform_random_policy(...) build these for you.

You are given one helper, `q_value` (below): the one-step lookahead q(s, a). It is
the piece that shows up inside every Bellman backup in the notes, so you will
probably want it in most of the functions.

Run  python utils/check.py  after each function to see how you are doing.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np


# ---------------------------------------------------------------------------
#  A helper you are given: the one-step lookahead q(s, a).
# ---------------------------------------------------------------------------
def q_value(env, V: np.ndarray, s: int, a: str, gamma: float) -> float:
    """The action value q(s, a): expected immediate reward plus discounted value
    of where you land, averaged over the environment's outcomes."""
    total = 0.0
    for prob, s_next, reward in env.transitions(s, a):
        total += prob * (reward + gamma * V[s_next])
    return total


# ===========================================================================
#  1. POLICY EVALUATION  (prediction) -- Week 2 notes, Chapter 2.
# ===========================================================================
def policy_evaluation(env, policy: Dict[int, Dict[str, float]], gamma: float,
                      theta: float = 1e-6) -> np.ndarray:
    """Return the value function V (a 1-D array of length env.n_states) of the
    given `policy`. `theta` is the stopping tolerance."""
    V=np.zeros(env.n_states)
    while True:
        delta=0.0
        V_new=np.copy(V)
        for s in env.states:
            if env.is_terminal(s):
                continue
            v=0.0
            for action,action_prob in policy[s].items():
                v+=action_prob*q_value(env,V,s,action,gamma)
            delta=max(delta,abs(v-V[s]))

            V_new[s]=v

        V=V_new
        if delta<theta:
            break
    return V   


# ===========================================================================
#  2. GREEDY POLICY  (the improvement step) -- Week 2 notes, Chapter 3.
# ===========================================================================
def greedy_policy(env, V: np.ndarray, gamma: float) -> Dict[int, Dict[str, float]]:
    """Return the deterministic policy that is greedy with respect to V, in the
    form {state: {action: 1.0}} for every non-terminal state."""
    policy = {}
    for s in env.states:
        
        if env.is_terminal(s):
            continue
        best_action = None
        best_q = -float('inf')
        
        for a in env.actions:
            q = q_value(env, V, s, a, gamma)
            if q > best_q:
                best_q = q
                best_action = a
                
        policy[s] = {best_action: 1.0}
    return policy



# ===========================================================================
#  3. POLICY ITERATION  (control) -- Week 2 notes, Chapter 3.
# ===========================================================================
def policy_iteration(env, gamma: float,
                     theta: float = 1e-6) -> Tuple[np.ndarray, Dict, int]:
    """Return (V, policy, n_iterations): the optimal value function, the optimal
    policy, and the number of policy-iteration steps taken."""
    #random
    
    policy = {}
    for s in env.states:
        if not env.is_terminal(s):
            
            policy[s] = {a: 0.25 for a in env.actions}
    n_iterations=0
    while True:
        V=policy_evaluation(env,policy,gamma,theta)

        new_policy=greedy_policy(env,V,gamma)
        n_iterations+=1

        if policy==new_policy:
            break
        policy=new_policy

    return V,policy,n_iterations
        




# ===========================================================================
#  4. VALUE ITERATION  (control) -- Week 2 notes, Chapter 4.
# ===========================================================================
def value_iteration(env, gamma: float,
                    theta: float = 1e-6) -> Tuple[np.ndarray, Dict, int]:
    """Return (V, policy, n_sweeps): the optimal value function, its greedy
    policy, and the number of sweeps taken. `theta` is the stopping tolerance."""
    V=np.zeros(env.n_states)
    n_iterations=0
    
    while True:
        delta=0.0
        V_new=np.copy(V)

        for s in env.states:
            if env.is_terminal(s):
                continue
            best_q=-float('inf')
            for a in env.actions:
                q=q_value(env,V,s,a,gamma)
                if q>best_q:
                    best_q=q
            delta = max(delta, abs(best_q - V[s]))
            V_new[s] = best_q
            
        V = V_new
        n_iterations += 1
        
        if delta < theta:
            break
            
    
    policy = greedy_policy(env, V, gamma)
    
    return V, policy, n_iterations



# ===========================================================================
#  5. VALUE ITERATION, IN PLACE  (asynchronous) -- Week 2 notes, Chapter 5.
# ===========================================================================
def value_iteration_inplace(env, gamma: float,
                            theta: float = 1e-6) -> Tuple[np.ndarray, Dict, int]:
    """Return (V, policy, n_sweeps), the same result as value_iteration but using
    in-place (asynchronous) updates."""
    V = np.zeros(len(env.states))
    n_sweeps = 0
    
    while True:
        delta = 0.0  
        for s in env.states:
            if env.is_terminal(s):
                continue
                
            v_old = V[s]
            best_q = -float('inf')
            
            for a in env.actions:
                q = q_value(env, V, s, a, gamma)
                if q > best_q:
                    best_q = q
                    
            V[s] = best_q
            delta = max(delta, abs(v_old - V[s]))
            
        n_sweeps += 1
        if delta < theta:
            break
            
    policy = greedy_policy(env, V, gamma)
    return V, policy, n_sweeps
   
# ===========================================================================
#  6. MODIFIED POLICY ITERATION  (the dial between VI and PI)
#     Week 2 notes, Chapter 5. `k` is how many evaluation sweeps you run per
#     improvement step.
# ===========================================================================
def modified_policy_iteration(env, gamma: float, k: int, theta: float = 1e-6):
    """Return (V, policy, n_iterations). `theta` is the stopping tolerance."""
    V = np.zeros(len(env.states))
    policy = {}
    for s in env.states:
        if not env.is_terminal(s):
            policy[s] = {a: 1.0 / len(env.actions) for a in env.actions}
            
    n_iterations = 0
    
    while True:
        
        V_old = np.copy(V) 
        
    
        for _ in range(k):
            V_new = np.copy(V)
            for s in env.states:
                if env.is_terminal(s):
                    continue
                
                v_s = 0
                for a, prob in policy[s].items():
                    v_s += prob * q_value(env, V, s, a, gamma)
                    
                V_new[s] = v_s
            V = V_new
            
     
        policy = greedy_policy(env, V, gamma)
        n_iterations += 1
        
        
        if np.max(np.abs(V - V_old)) < theta:
            break
            
    return V, policy, n_iterations
   