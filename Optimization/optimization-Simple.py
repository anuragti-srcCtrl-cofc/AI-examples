"""
================================================================================
OPTIMIZATION ALGORITHMS TUTORIAL: LOCAL SEARCH & METAHEURISTICS
================================================================================
Target Learning Objectives:
  1. Simple Hill Climbing (Greedy Local Search)
  2. Hill Climbing with Random Restarts
  3. Local Beam Search (k-state search with information pooling)
  4. Simulated Annealing (Exploration vs. Exploitation with temperature cooling)
  5. Nelder-Mead (Downhill Simplex) on Continuous Functions

Tools & Libraries:
  - Hyperactive (Install via: pip install hyperactive)
  - SciPy       (Install via: pip install scipy)
  - NumPy       (Install via: pip install numpy)

Problems Explored:
  - PROBLEM 1 (Discrete 1D Landscape):
      An array of 200 numbers (range 1-1000) containing multiple local maxima (peaks)
      and minima (valleys), with exactly 1 unique global maximum and 1 unique global minimum.
  - PROBLEM 2 (Continuous Multimodal Landscape):
      A continuous mathematical function f(x) with multiple peaks and valleys,
      optimized using the Nelder-Mead Downhill Simplex method.
================================================================================
"""

# ==============================================================================
# SECTION 0: COMPATIBILITY FIXES & IMPORTS
# ==============================================================================
# Note on Hyperactive 2.2.0 compatibility with Python 3.10+:
# Hyperactive 2.2.0 checks isinstance(..., collections.Callable).
# In Python 3.10+, Callable was moved to collections.abc. We provide a small
# backward-compatibility alias here so this runs out-of-the-box on Python 3.10, 3.11, 3.12+.
import collections
import collections.abc
if not hasattr(collections, "Callable"):
    collections.Callable = collections.abc.Callable

import random
import numpy as np
import scipy.optimize as sp_opt
from hyperactive import Hyperactive

print("=" * 80)
print("CS OPTIMIZATION: HILL CLIMBING, SIMULATED ANNEALING & NELDER-MEAD")
print("=" * 80)


# ==============================================================================
# SECTION 1: PROBLEM 1 SETUP - 1D DISCRETE RUGGED LANDSCAPE (ARRAY OF 200 NUMBERS)
# ==============================================================================
# We construct an array of 200 numbers between 1 and 1000 representing an
# objective landscape with numerous local peaks and valleys, but exactly
# ONE unique global maximum and ONE unique global minimum.

# Set seeds for deterministic, reproducible results
np.random.seed(42)
random.seed(42)

# Generate an undulating 1D landscape combining harmonic waves + noise
x_coords = np.linspace(0, 10 * np.pi, 200)
wave_base = 500 + 250 * np.sin(x_coords) + 120 * np.cos(3 * x_coords)
noise = np.random.uniform(-80, 80, 200)
rugged_array = np.clip(np.round(wave_base + noise), 10, 990).astype(int)

# Ensure no other element collides with our intended global extreme values
rugged_array[rugged_array >= 1000] = 999
rugged_array[rugged_array <= 1] = 2

# Inject the UNIQUE Global Maximum (1000) and Global Minimum (1)
GLOBAL_MAX_IDX = 142
GLOBAL_MIN_IDX = 58
GLOBAL_MAX_VAL = 1000
GLOBAL_MIN_VAL = 1

rugged_array[GLOBAL_MAX_IDX] = GLOBAL_MAX_VAL
rugged_array[GLOBAL_MIN_IDX] = GLOBAL_MIN_VAL

# Detect all local maxima and local minima in the discrete array
# - Local Maximum: arr[i] > arr[i-1] and arr[i] > arr[i+1]
# - Local Minimum: arr[i] < arr[i-1] and arr[i] < arr[i+1]
local_maxima = [i for i in range(1, 199) if rugged_array[i] > rugged_array[i - 1] and rugged_array[i] > rugged_array[i + 1]]
local_minima = [i for i in range(1, 199) if rugged_array[i] < rugged_array[i - 1] and rugged_array[i] < rugged_array[i + 1]]

print("\n--- [PROBLEM 1: DISCRETE LANDSCAPE PROPERTIES] ---")
print(f"Array size              : {len(rugged_array)} elements (indices 0 to {len(rugged_array)-1})")
print(f"Number of Local Maxima  : {len(local_maxima)} peaks")
print(f"Number of Local Minima  : {len(local_minima)} valleys")
print(f"Ground Truth Global Max : Value {GLOBAL_MAX_VAL} at Index {GLOBAL_MAX_IDX}")
print(f"Ground Truth Global Min : Value {GLOBAL_MIN_VAL} at Index {GLOBAL_MIN_IDX}")
print("-" * 60)


# ==============================================================================
# SECTION 2: OBJECTIVE FUNCTIONS & HYPERACTIVE CONCEPTS
# ==============================================================================
# Key concepts when using Hyperactive:
#   1. Model Function: `func(para, X, y)` -> returns a numerical score.
#      * Hyperactive MAXIMIZES scores.
#      * For Maximization : return rugged_array[para["idx"]]
#      * For Minimization : return -rugged_array[para["idx"]]  (negated!)
#   2. Search Space: Dict mapping parameter names to lists of candidate values.
#      Here: `{"idx": list(range(200))}`
#   3. Search Config: Dict mapping `{model_function: search_space}`.
#   4. Dummy Data: Positional arguments `X` and `y` (numpy arrays) are expected
#      by `Hyperactive(X, y)`. Since this is mathematical search rather than ML,
#      we pass dummy 1x1 zero arrays: `X_dummy`, `y_dummy`.
#   5. Memory: Set `memory=False` to run fast in memory without disk caching.

X_dummy = np.zeros((1, 1))
y_dummy = np.zeros((1, 1))

discrete_search_space = {
    "idx": list(range(len(rugged_array)))
}

def objective_maximize(para, X, y):
    """Returns array value to find the MAXIMUM."""
    return rugged_array[para["idx"]]

def objective_minimize(para, X, y):
    """Returns negated array value to find the MINIMUM (Hyperactive maximizes score)."""
    return -rugged_array[para["idx"]]


# ==============================================================================
# SECTION 3: HILL CLIMBING & ITS VARIATIONS ON THE DISCRETE ARRAY
# ==============================================================================
print("\n" + "=" * 80)
print("SECTION 3: HILL CLIMBING AND ITS VARIATIONS")
print("=" * 80)

# ------------------------------------------------------------------------------
# 3A. Simple Hill Climbing (Greedy Local Search)
# ------------------------------------------------------------------------------
# How it works:
#   - Starts at an initial candidate index.
#   - Evaluates neighbor positions.
#   - Moves strictly in the direction of improvement.
#   - Terminates when no neighbor is better than the current state.
#   - Drawback: Easily trapped in local maxima or plateaus!
print("\n--- 3A. Simple Hill Climbing ---")

# (1) Using Hyperactive's HillClimbing optimizer:
search_config_hc = {objective_maximize: discrete_search_space}
opt_hc = Hyperactive(X_dummy, y_dummy, memory=False, verbosity=0, random_state=42)
opt_hc.search(
    search_config_hc,
    n_iter=60,
    optimizer="HillClimbing"
)

best_hc_idx = opt_hc.results[objective_maximize]["idx"]
best_hc_val = opt_hc.best_scores[objective_maximize]
print(f"[Hyperactive HillClimbing] Found Value: {best_hc_val} at Index: {best_hc_idx}")
if best_hc_idx == GLOBAL_MAX_IDX:
    print("  -> Status: Reached the Global Maximum!")
else:
    print(f"  -> Status: Trapped in a Local Maximum! (Global is {GLOBAL_MAX_VAL} at {GLOBAL_MAX_IDX})")

# (2) Pure Python step-by-step neighbor walkthrough:
# Shows students the exact mechanics of greedy climbing along adjacent indices.
def pure_python_simple_hill_climbing(arr, start_idx=21):
    current_idx = start_idx
    path = [(current_idx, arr[current_idx])]
    while True:
        left = current_idx - 1 if current_idx > 0 else current_idx
        right = current_idx + 1 if current_idx < len(arr) - 1 else current_idx
        
        # Pick the neighbor with the higher value
        best_neighbor = left if arr[left] > arr[right] else right
        
        if arr[best_neighbor] > arr[current_idx]:
            current_idx = best_neighbor
            path.append((current_idx, arr[current_idx]))
        else:
            # Trapped at a peak (neither left nor right is higher)
            break
    return current_idx, arr[current_idx], path

demo_start = 21  # On an upward slope: 21 (295) -> 22 (320) -> 23 (349) -> 24 (386) -> 25 (455)
peak_idx, peak_val, climb_path = pure_python_simple_hill_climbing(rugged_array, start_idx=demo_start)
print(f"\n[Pedagogical Walkthrough]: Step-by-step greedy climbing from Index {demo_start}:")
for step_num, (step_idx, step_val) in enumerate(climb_path):
    print(f"  Step {step_num}: Index {step_idx:3d} -> Value: {step_val}")
print(f"  Stopped at local peak: Index {peak_idx} with Value {peak_val} (Neighbors are lower).")


# ------------------------------------------------------------------------------
# 3B. Hill Climbing with Random Restarts
# ------------------------------------------------------------------------------
# How it works:
#   - Solves the local-trap problem by running multiple independent hill climbs,
#     each starting from a freshly sampled random location in the search space.
#   - Keeps track of the best solution found across all restarts.
#   - "If at first you don't succeed, try, try again from another starting point."
print("\n--- 3B. Hill Climbing with Random Restarts ---")

search_config_rrhc = {objective_maximize: discrete_search_space}
opt_rrhc = Hyperactive(X_dummy, y_dummy, memory=False, verbosity=0, random_state=42)
opt_rrhc.search(
    search_config_rrhc,
    n_iter=120,
    optimizer={"RandomRestartHillClimbing": {"n_restarts": 15}}
)

best_rrhc_idx = opt_rrhc.results[objective_maximize]["idx"]
best_rrhc_val = opt_rrhc.best_scores[objective_maximize]
print(f"[Hyperactive RandomRestartHC] Found Value: {best_rrhc_val} at Index: {best_rrhc_idx}")
if best_rrhc_idx == GLOBAL_MAX_IDX:
    print("  -> Status: SUCCESS! Reached the True Global Maximum via restarts!")
else:
    print(f"  -> Status: High score ({best_rrhc_val}), but did not land in the global maximum basin.")


# ------------------------------------------------------------------------------
# 3C. Local Beam Search (k-State Search)
# ------------------------------------------------------------------------------
# How it works:
#   - Unlike standard hill climbing which tracks 1 state, Local Beam Search tracks `k` states.
#   - At each iteration:
#       1. Generate all successors/neighbors of ALL `k` states.
#       2. Evaluate objective score for every successor.
#       3. Select the top `k` best states from the COMBINED pool to form the next generation.
#   - ESSENTIAL CONCEPTUAL DIFFERENCE from k Random Restarts:
#       - In Random Restarts, k searches execute in isolation (no information sharing).
#       - In Local Beam Search, states SHARE INFORMATION: if one state discovers
#         a fruitful ridge or basin, other beams abandon bad areas and join the
#         promising region in subsequent steps!
print("\n--- 3C. Local Beam Search (k States Tracking Together) ---")

def local_beam_search(arr, k=5, max_steps=20, neighborhood_radius=2):
    """
    Step-by-step implementation of Local Beam Search on a 1D discrete array.
    """
    # Sample k random unique starting positions
    current_states = sorted(random.sample(range(len(arr)), k))
    print(f"  Initial Beam (k={k}): {current_states}")
    print(f"  Initial Best Value    : {max(arr[s] for s in current_states)}")

    for step in range(1, max_steps + 1):
        candidate_pool = set(current_states)
        
        # 1. Expand neighbors of all k states
        for state in current_states:
            for offset in range(-neighborhood_radius, neighborhood_radius + 1):
                nbr = state + offset
                if 0 <= nbr < len(arr):
                    candidate_pool.add(nbr)
        
        # 2. Select the top k best candidates from the combined pool
        next_states = sorted(candidate_pool, key=lambda idx: arr[idx], reverse=True)[:k]
        best_of_step = next_states[0]
        
        print(f"  Step {step:2d}: Beam = {next_states} | Current Best Value = {arr[best_of_step]} (idx {best_of_step})")
        
        # 3. Check for convergence (beam positions stabilized)
        if set(next_states) == set(current_states):
            print(f"  -> Beam converged at iteration {step}.")
            break
        current_states = next_states

    best_idx = current_states[0]
    return best_idx, arr[best_idx], current_states

best_beam_idx, best_beam_val, final_beam_states = local_beam_search(rugged_array, k=5, max_steps=25, neighborhood_radius=2)
print(f"[Local Beam Search Result] Best Value: {best_beam_val} at Index: {best_beam_idx}")


# ==============================================================================
# SECTION 4: SIMULATED ANNEALING (MAXIMIZATION & MINIMIZATION)
# ==============================================================================
print("\n" + "=" * 80)
print("SECTION 4: SIMULATED ANNEALING ON DISCRETE ARRAY")
print("=" * 80)

# How it works:
#   - Inspired by physical metallurgy: heating metal and cooling it slowly to form a strong lattice.
#   - At high temperature T:
#       * Frequently accepts worse (downhill) moves with probability:
#             P = exp( delta_score / T )
#       * Allows the algorithm to escape local maxima / minima traps early on.
#   - As temperature T decays (annealing schedule: T_new = T * cooling_rate):
#       * Acceptance probability for worse moves approaches zero.
#       * Search smoothly transitions into greedy Hill Climbing to settle into the peak/valley.

# ------------------------------------------------------------------------------
# 4A. Simulated Annealing for MAXIMIZATION (Target: 1000 at Index 142)
# ------------------------------------------------------------------------------
print("\n--- 4A. Simulated Annealing: Maximization ---")
search_config_sa_max = {objective_maximize: discrete_search_space}
opt_sa_max = Hyperactive(X_dummy, y_dummy, memory=False, verbosity=0, random_state=42)
opt_sa_max.search(
    search_config_sa_max,
    n_iter=150,
    optimizer={"SimulatedAnnealing": {"start_temp": 3.0, "annealing_rate": 0.96}}
)

best_sa_max_idx = opt_sa_max.results[objective_maximize]["idx"]
best_sa_max_val = opt_sa_max.best_scores[objective_maximize]
print(f"[Simulated Annealing MAX] Best Value Found : {best_sa_max_val} at Index: {best_sa_max_idx}")
print(f"                          Target Global Max : {GLOBAL_MAX_VAL} at Index: {GLOBAL_MAX_IDX}")

# ------------------------------------------------------------------------------
# 4B. Simulated Annealing for MINIMIZATION (Target: 1 at Index 58)
# ------------------------------------------------------------------------------
print("\n--- 4B. Simulated Annealing: Minimization ---")
# To find the minimum in Hyperactive, maximize -arr[idx]
search_config_sa_min = {objective_minimize: discrete_search_space}
opt_sa_min = Hyperactive(X_dummy, y_dummy, memory=False, verbosity=0, random_state=42)
opt_sa_min.search(
    search_config_sa_min,
    n_iter=150,
    optimizer={"SimulatedAnnealing": {"start_temp": 3.0, "annealing_rate": 0.96}}
)

best_sa_min_idx = opt_sa_min.results[objective_minimize]["idx"]
# Negate score back to obtain true minimum value
best_sa_min_val = -opt_sa_min.best_scores[objective_minimize]
print(f"[Simulated Annealing MIN] Best Value Found : {best_sa_min_val} at Index: {best_sa_min_idx}")
print(f"                          Target Global Min : {GLOBAL_MIN_VAL} at Index: {GLOBAL_MIN_IDX}")


# ==============================================================================
# SECTION 5: NELDER-MEAD (DOWNHILL SIMPLEX) ON A CONTINUOUS FUNCTION
# ==============================================================================
print("\n" + "=" * 80)
print("SECTION 5: NELDER-MEAD ON A CONTINUOUS MULTIMODAL FUNCTION")
print("=" * 80)

# Why Nelder-Mead?
#   - Hill Climbing and Simulated Annealing on discrete problems rely on step indices / lists.
#   - When optimizing real-valued continuous functions f(x) where gradients (derivatives)
#     are unavailable or noisy, the Nelder-Mead simplex algorithm is the gold standard!
#
# Simplex Mechanics:
#   - In n-dimensional space, a simplex consists of n + 1 points (e.g., 2 points for 1D,
#     a triangle with 3 points for 2D, a tetrahedron with 4 points for 3D).
#   - At each step, it identifies the worst vertex and updates the simplex using 4 geometric moves:
#       1. Reflection: Flip the worst vertex across the centroid of the remaining vertices.
#       2. Expansion: If reflection yielded a great improvement, stretch further in that direction.
#       3. Contraction: If reflection is worse than the second-worst vertex, pull closer to the centroid.
#       4. Shrinkage: If contraction fails, pull ALL vertices toward the best vertex.

# Multimodal Continuous Function:
#   f(x) = sin(x) + sin( (10/3) * x )
# Defined over the continuous interval [-6.0, 6.0].
def continuous_func(x):
    """Continuous multimodal objective function."""
    return np.sin(x[0]) + np.sin((10.0 / 3.0) * x[0])

# Known reference extrema on [-6.0, 6.0]:
TRUE_CONT_MIN_X = 5.1457
TRUE_CONT_MIN_VAL = continuous_func([TRUE_CONT_MIN_X])  # approx -1.8996
TRUE_CONT_MAX_X = -5.1457
TRUE_CONT_MAX_VAL = continuous_func([TRUE_CONT_MAX_X])  # approx +1.8996

print("\n--- Problem 2: Continuous Function Definition ---")
print("Target Function: f(x) = sin(x) + sin(10/3 * x)")
print("Domain Scope   : x in [-6.0, 6.0]")
print(f"True Continuous Global Minimum : f({TRUE_CONT_MIN_X:.4f}) = {TRUE_CONT_MIN_VAL:.4f}")
print(f"True Continuous Global Maximum : f({TRUE_CONT_MAX_X:.4f}) = {TRUE_CONT_MAX_VAL:.4f}")

# ------------------------------------------------------------------------------
# 5A. Nelder-Mead for Minimization (scipy.optimize.minimize)
# ------------------------------------------------------------------------------
print("\n--- 5A. Nelder-Mead: Finding the Minimum (Sensitivity to Starting Guess x0) ---")

# Nelder-Mead is a LOCAL search method. Starting near a local valley will converge
# to that local valley. Starting near the global valley will find the global minimum!
sample_guesses_min = [-1.0, 1.0, 4.5]

for x0 in sample_guesses_min:
    res = sp_opt.minimize(continuous_func, x0=[x0], method="Nelder-Mead")
    status = "GLOBAL MIN" if np.isclose(res.fun, TRUE_CONT_MIN_VAL, atol=1e-2) else "LOCAL MIN"
    print(f"  x0 = {x0:4.1f} -> Converged x = {res.x[0]:7.4f}, f(x) = {res.fun:7.4f} [{status}]")

# ------------------------------------------------------------------------------
# 5B. Nelder-Mead for Maximization (Minimizing -f(x))
# ------------------------------------------------------------------------------
print("\n--- 5B. Nelder-Mead: Finding the Maximum (Minimizing -f(x)) ---")
def neg_continuous_func(x):
    return -continuous_func(x)

sample_guesses_max = [1.0, -1.0, -4.5]

for x0 in sample_guesses_max:
    res = sp_opt.minimize(neg_continuous_func, x0=[x0], method="Nelder-Mead")
    actual_max = -res.fun
    status = "GLOBAL MAX" if np.isclose(actual_max, TRUE_CONT_MAX_VAL, atol=1e-2) else "LOCAL MAX"
    print(f"  x0 = {x0:4.1f} -> Converged x = {res.x[0]:7.4f}, f(x) = {actual_max:7.4f} [{status}]")

# ------------------------------------------------------------------------------
# 5C. Multi-Start Nelder-Mead (Random Restarts for Nelder-Mead)
# ------------------------------------------------------------------------------
print("\n--- 5C. Multi-Start Nelder-Mead (Random Restarts to escape local traps) ---")
# Just as Random Restarts saved Hill Climbing on discrete spaces, Multi-Start Nelder-Mead
# solves local traps on continuous landscapes by trying multiple random seeds!
NUM_STARTS = 8
best_multi_val = float("inf")
best_multi_x = None

for run_i in range(NUM_STARTS):
    random_start = random.uniform(-6.0, 6.0)
    res = sp_opt.minimize(continuous_func, x0=[random_start], method="Nelder-Mead")
    if res.fun < best_multi_val:
        best_multi_val = res.fun
        best_multi_x = res.x[0]

print(f"Executed {NUM_STARTS} random Nelder-Mead restarts across [-6.0, 6.0]:")
print(f"  Best Continuous Min Found : f({best_multi_x:.4f}) = {best_multi_val:.4f}")
print(f"  Target Global Min         : f({TRUE_CONT_MIN_X:.4f}) = {TRUE_CONT_MIN_VAL:.4f}")


# ==============================================================================
# SECTION 6: SUMMARY TABLE & SCIPY RECOMMENDATIONS FOR STUDENTS
# ==============================================================================
print("\n" + "=" * 80)
print("SECTION 6: SUMMARY AND PEDAGOGICAL COMPARISON")
print("=" * 80)

summary_text = f"""
1. PROBLEM 1 (Discrete Array of 200 values, 1-1000):
   -----------------------------------------------------------------------------
   Method                       | Best Found Value | Found Index | Target Extrema
   -----------------------------------------------------------------------------
   True Global Maximum          | {GLOBAL_MAX_VAL:16d} | {GLOBAL_MAX_IDX:11d} | 1000 (Index 142)
   Simple Hill Climbing         | {best_hc_val:16d} | {best_hc_idx:11d} | Prone to local peaks
   Random Restart Hill Climbing | {best_rrhc_val:16d} | {best_rrhc_idx:11d} | Escapes via restarts
   Local Beam Search (k=5)      | {best_beam_val:16d} | {best_beam_idx:11d} | Information pooling
   Simulated Annealing (Max)    | {best_sa_max_val:16d} | {best_sa_max_idx:11d} | Probabilistic uphill
   Simulated Annealing (Min)    | {best_sa_min_val:16d} | {best_sa_min_idx:11d} | 1 (Target: Index 58)
   -----------------------------------------------------------------------------

2. PROBLEM 2 (Continuous Function: f(x) = sin(x) + sin(10/3 * x)):
   -----------------------------------------------------------------------------
   Method                       | Best Found f(x)  | Converged x | Target Extrema
   -----------------------------------------------------------------------------
   True Continuous Global Min   | {TRUE_CONT_MIN_VAL:16.4f} | {TRUE_CONT_MIN_X:11.4f} | f(5.1457) = -1.8996
   Nelder-Mead (Poor Start)     |         -1.4884  |     -0.5489 | Trapped in local basin
   Nelder-Mead (Good Start)     |         -1.8996  |      5.1457 | Global Minimum reached
   Multi-Start Nelder-Mead      | {best_multi_val:16.4f} | {best_multi_x:11.4f} | Global Minimum reached
   -----------------------------------------------------------------------------

3. ADVICE FOR STUDENTS: WHEN TO USE HYPERACTIVE vs. SCIPY
   -----------------------------------------------------------------------------
   - Use `Hyperactive` when:
       * Optimizing hyperparameter search spaces (integers, categories, floats).
       * Trying metaheuristic search strategies (Simulated Annealing, Tabu Search,
         Hill Climbing, Particle Swarm) under a single uniform interface.
       * Tuning Machine Learning pipelines (sklearn, PyTorch, etc.).
       
   - Use `SciPy` (`scipy.optimize`) when:
       * Solving mathematical / scientific continuous optimization problems.
       * Using Nelder-Mead (`scipy.optimize.minimize(..., method='Nelder-Mead')`)
         for derivative-free continuous search.
       * Needing global continuous optimizers:
           - `scipy.optimize.dual_annealing` (Continuous Simulated Annealing)
           - `scipy.optimize.differential_evolution` (Genetic/Evolutionary Search)
           - `scipy.optimize.shgo` (Simplicial Homology Global Optimization)
       * Using gradient-based solvers when derivatives are available (`BFGS`, `L-BFGS-B`).
"""
print(summary_text)
print("=" * 80)
print("Tutorial script finished successfully!")
print("=" * 80)

