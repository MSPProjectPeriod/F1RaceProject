# --- Timing: Start timer at script start ---
import os
import pickle
import time

# Start timer for total runtime measurement
start_time = time.time()
# === F1 Race Strategy Optimizer ===
# === F1 Race Strategy Optimizer ===
# This script simulates a race strategy using three stint functions and two pit stops.

import pandas as pd

# --- Precomputed time cache (global) ---
precomputed_time_cache = {}
# Try to load cache from disk if exists
if os.path.exists('precomputed_time_cache.pkl'):
    with open('precomputed_time_cache.pkl', 'rb') as f:
        precomputed_time_cache = pickle.load(f)

# --- Global Configuration Variables ---
number_of_laps = 100             # Total number of laps in the race (max lap limit)
average_pit_time = 30.0          # Default average pit stop time (seconds)
min_lap_buffer = 5               # Laps to avoid at start/end before a pit may occur
min_lap_between_pits = 5         # Minimum laps between two pit stops

# Stint limits for each function: [soft_limit, hard_limit]
stint_limits_array = [
    [20, 30],   # Stint 1 [soft, hard]
    [65, 75],   # Stint 2 [soft, hard]
    [18, 28],   # Stint 3 [soft, hard]
]


# Penalty growth factors for each stint function (by index)
penalty_growth_factors = [
    0.5,  # Stint 1 penalty factor
    0.4,  # Stint 2 penalty factor
    0.6,  # Stint 3 penalty factor
]

 # --- Fast/Slow Stint Settings ---
# Mode adjustment for stint expressions using a global alpha
mode_alpha = 0.2  # Controls how aggressive/durable behavior affects lap time

# Define mode adjustment functions per stint function
from sympy import diff




double_check_comparison = False   # Enable detailed comparison logging if True
best_combo_info = None            # Stores optimal pit stop info for later visualization

# --- Imports ---
from sympy import symbols, sympify, integrate, lambdify
import matplotlib.pyplot as plt
import numpy as np

# Symbolic variables
x = symbols('x')                 # Primary lap counter for each stint
u = symbols('u')                 # Reset lap counter after a pit stop

import sys

# --- Load stint functions from CSV ---
data_file = "csv_files/csv_results_0_2024_Bahrain_Grand_Prix_Grand_Prix.csv"
df = pd.read_csv(data_file)

print("\nAvailable drivers and data types:")
print(df['driver'].unique())
driver_filter = input("Enter driver name or leave blank to use average data: ").strip().lower()

if driver_filter:
    filtered_df = df[df['driver'].str.lower() == driver_filter]
    if not filtered_df.empty:
        number_of_laps = int(filtered_df['Total Laps'].iloc[0])
    else:
        print(f"Driver '{driver_filter}' not found. Using average data.")
        filtered_df = df[df['driver'].str.upper().str.startswith("AVERAGE_EQUATION")]
else:
    filtered_df = df[df['driver'].str.upper().str.startswith("AVERAGE_EQUATION")]
    number_of_laps = int(df['Total Laps'].iloc[0])

# Extract average equations by compound
compound_map = {'Soft': None, 'Medium': None, 'Hard': None}
for compound in compound_map.keys():
    match = filtered_df[
        (filtered_df['tiretype'].str.upper() == compound.upper())
    ]
    if not match.empty:
        compound_map[compound] = match.iloc[0]['besttrendline']

# After mapping, print warning for missing formulas
for compound, formula in compound_map.items():
    if formula is None:
        print(f"⚠️ Missing formula for: {compound}")

# Map compound functions
from sympy.parsing.sympy_parser import parse_expr

def parse_equation_string(expr_str):
    try:
        expr_str = expr_str.replace("^", "**")
        expr_str = expr_str.replace("e**", "exp(1)**")
        expr_str = expr_str.replace("e*", "exp(1)*")
        return parse_expr(expr_str, evaluate=True)
    except Exception as e:
        print(f"Failed to parse equation: {expr_str}")
        raise e

tire_exprs = {}
if compound_map['Soft']:
    tire_exprs['soft'] = parse_equation_string(compound_map['Soft'])
if compound_map['Medium']:
    tire_exprs['medium'] = parse_equation_string(compound_map['Medium'])
if compound_map['Hard']:
    tire_exprs['hard'] = parse_equation_string(compound_map['Hard'])

# Assign only available stint functions in preferred order: Soft, Medium, Hard
stint_exprs = []
if 'soft' in tire_exprs:
    stint_exprs.append(tire_exprs['soft'])
if 'medium' in tire_exprs:
    stint_exprs.append(tire_exprs['medium'])
if 'hard' in tire_exprs:
    stint_exprs.append(tire_exprs['hard'])

if len(stint_exprs) < 2:
    raise ValueError("At least two valid tire performance functions are required. Only found: " + ', '.join(tire_exprs.keys()))

print(f"✔️ Using tire compounds: {list(tire_exprs.keys())}")

# Fallback: Assign symbolic names
stint1_func, stint2_func, stint3_func = (stint_exprs + [stint_exprs[-1]] * 3)[:3]


# Move degradation_factors definition here before use
degradation_factors = {
    'neutral': 1.0,
    'aggressive': 1.1,
    'durable': 0.9
}

print("Loaded Compound Functions:")
if 'soft' in tire_exprs:
    print("Soft Compound Function:", tire_exprs['soft'])
if 'medium' in tire_exprs:
    print("Medium Compound Function:", tire_exprs['medium'])
if 'hard' in tire_exprs:
    print("Hard Compound Function:", tire_exprs['hard'])


# Map index to tire type and compound from stint_exprs


print(f"\nTotal Number of Laps in Race: {number_of_laps}")



# --- Derived tire max values per mode ---

import math

# --- Load and store soft/hard tire limits per compound for future use ---
compound_limits = {}
for expr in [stint1_func, stint2_func, stint3_func]:
    for label, formula in tire_exprs.items():
        if expr == formula:
            match = filtered_df[
                (filtered_df['tiretype'].str.lower() == label)
            ]
            if not match.empty:
                compound = match.iloc[0]['compound']
                soft_limit = match.iloc[0]['compound_avg']
                hard_limit = match.iloc[0]['compound_max']
                compound_limits[label] = {'compound': compound, 'soft_limit': soft_limit, 'hard_limit': hard_limit}
            break

# --- Derived tire max values per mode ---
tire_max_limits = {}
for compound, limits in compound_limits.items():
    soft_limit = limits['soft_limit']
    hard_limit = limits['hard_limit']
    tire_max_limits[compound] = {
        'durable': {
            'soft_limit': math.ceil(soft_limit / degradation_factors['durable']),
            'hard_limit': math.ceil(hard_limit / degradation_factors['durable'])
        },
        'neutral': {
            'soft_limit': math.ceil(soft_limit / degradation_factors['neutral']),
            'hard_limit': math.ceil(hard_limit / degradation_factors['neutral'])
        },
        'aggressive': {
            'soft_limit': math.ceil(soft_limit / degradation_factors['aggressive']),
            'hard_limit': math.ceil(hard_limit / degradation_factors['aggressive'])
        }
    }

# --- Precompute allowed modes for each (compound, stint_len) pair ---
allowed_mode_cache = {}
for compound in tire_max_limits:
    for stint_len in range(1, number_of_laps + 1):
        allowed = []
        for mode in ['durable', 'neutral', 'aggressive']:
            if stint_len <= tire_max_limits[compound][mode]['hard_limit']:
                allowed.append(mode)
        allowed_mode_cache[(compound, stint_len)] = allowed


print("\n--- Tire Max Limits by Mode (Rounded) ---")
for compound, modes in tire_max_limits.items():
    soft_d = modes['durable']['soft_limit']
    hard_d = modes['durable']['hard_limit']
    soft_n = modes['neutral']['soft_limit']
    hard_n = modes['neutral']['hard_limit']
    soft_a = modes['aggressive']['soft_limit']
    hard_a = modes['aggressive']['hard_limit']
    print(f"{compound.capitalize()} Tire - Durable [SM:{soft_d}|HM:{hard_d}]  Neutral [SM:{soft_n}|HM:{hard_n}]  Aggressive [SM:{soft_a}|HM:{hard_a}]")











# --- Stint mode settings (moved here after stint1_func, stint2_func, stint3_func are defined) ---
stint_base_exprs = [stint1_func, stint2_func, stint3_func]

# === Global mode order/factor definitions ===
mode_order = ['aggressive', 'neutral', 'durable']

# --- Define stint_mode_settings before get_mode_func is used ---
stint_mode_settings = {}
for i, base_expr in enumerate(stint_base_exprs):
    stint_mode_settings[i] = {}
    for mode, scale in degradation_factors.items():
        adjusted_expr = base_expr.subs(x, x * scale)
        stint_mode_settings[i][mode] = adjusted_expr




# --- Helper: Map stint function to its index for efficient lookup ---
func_index_map = {}
for idx, func in enumerate([stint1_func, stint2_func, stint3_func]):
    func_index_map[func] = idx

# Initialize function cache if not already present
if 'func_mode_cache' not in globals():
    func_mode_cache = {}
def get_mode_func(f_idx, mode):
    if (f_idx, mode) not in func_mode_cache:
        adjusted_expr = stint_mode_settings[f_idx][mode]
        expr_num = adjusted_expr.subs({x: u})  # ensure all symbolic variables are replaced
        func_mode_cache[(f_idx, mode)] = lambdify(u, expr_num, modules="numpy")
    return func_mode_cache[(f_idx, mode)]

 # --- Precompute penalty cache ---
max_penalty_len = number_of_laps
penalty_cache = np.cumsum(np.exp(np.arange(1, max_penalty_len + 1) / 5) - 1)

# Fatigue penalty function (uses penalty_cache)
def fatigue_penalty(t_array, soft_limit, factor):
    return np.zeros_like(t_array, dtype=float)

results = []


# === Multi-Pit Stop Strategy Candidates ===
from itertools import combinations, product

print("\n--- Multi-Pit Stop Strategy Candidates ---")

from itertools import product as iter_product
from itertools import combinations_with_replacement

# Store results for each multi-pit strategy



for num_pits in [1, 2, 3, 4]:
    print(f"\n== {num_pits}-Pit Strategy ==")
    strat_start_time = time.time()
    best_time = float('inf')
    best_plan = None
    best_plan_compounds = None

    num_stints = num_pits + 1
    possible_func_combos = []
    for func_combo in product(stint_exprs, repeat=num_stints):
        compounds_used = []
        for func in func_combo:
            for label, expr in tire_exprs.items():
                if expr == func:
                    compounds_used.append(label)
        if len(set(compounds_used)) >= 2:
            possible_func_combos.append(func_combo)

    # Generate all legal pit stop positions
    pit_positions = range(min_lap_buffer, number_of_laps - min_lap_buffer)
    for pit_combo in combinations(pit_positions, num_pits):
        # Check minimum laps between pits
        if any(pit_combo[i+1] - pit_combo[i] < min_lap_between_pits for i in range(len(pit_combo)-1)):
            continue
        # Stint lengths: first stint, then differences, then last stint
        stints = [pit_combo[0]] + [pit_combo[i+1] - pit_combo[i] for i in range(len(pit_combo)-1)] + [number_of_laps - pit_combo[-1]]
        if any(s <= 0 for s in stints):
            continue

        # Inserted: For 4-pit strategy, require exactly 5 stints
        if num_pits == 4 and len(stints) != 5:
            continue

        for func_combo in possible_func_combos:
            compounds = []
            func_indices = []
            for func in func_combo:
                for label, val in tire_exprs.items():
                    if val == func:
                        compounds.append(label)
                        break
                func_indices.append(func_index_map[func])
            if len(compounds) != len(stints) or len(func_indices) != len(stints):
                continue

            valid_modes = []
            for compound, stint_len in zip(compounds, stints):
                valid_modes.append(allowed_mode_cache.get((compound, stint_len), []))

            # If any stint has no valid modes, skip this pit combo
            if any(len(modes) == 0 for modes in valid_modes):
                continue

            # Skip if too many mode combinations
            if np.prod([len(m) for m in valid_modes]) > 200:
                continue

            for mode_combo in product(*valid_modes):
                total_time = 0
                valid = True
                for i, (compound, stint_len, mode, f_idx) in enumerate(zip(compounds, stints, mode_combo, func_indices)):
                    f = get_mode_func(f_idx, mode)
                    cache_key = (compound, mode, stint_len)
                    if cache_key in precomputed_time_cache:
                        stint_time = precomputed_time_cache[cache_key]
                    else:
                        laps = np.arange(stint_len)
                        penalty = fatigue_penalty(laps, stint_limits_array[f_idx][0], penalty_growth_factors[f_idx])
                        try:
                            stint_time = np.sum(f(laps) + penalty)
                        except Exception:
                            valid = False
                            break
                        precomputed_time_cache[cache_key] = stint_time
                    total_time += stint_time
                if not valid:
                    continue
                total_time += average_pit_time * num_pits
                if total_time < best_time:
                    best_time = total_time
                    best_plan = (func_combo, pit_combo, stints, mode_combo)
                    best_plan_compounds = compounds

    # Before appending, ensure best_plan is fully defined and formatted
    if best_plan:
        func_combo, pits, stints, modes = best_plan
        compounds = best_plan_compounds if best_plan_compounds is not None else []
        # For 4-pit strategy, print valid results if found (inserted block)
        if num_pits == 4:
            print(f"Best {num_pits}-stop strategy:")
            print(f"Pits at: {pits}, Stint Lengths: {stints}, Modes: {modes}")
            print(f"Tires: {compounds}")
            print(f"→ Total Race Time: {best_time:.2f}s")
        print(f"Best {num_pits}-stop strategy:")
        print(f"Pits at: {pits}, Stint Lengths: {stints}, Modes: {modes}")
        print(f"Tires: {compounds}")
        print(f"→ Total Race Time: {best_time:.2f}s")
        # Format for results: (func_combo, pits, stints, modes, best_time)
        formatted_plan = func_combo + (pits, stints, modes, best_time)
        results.append((num_pits, formatted_plan))
    else:
        results.append((num_pits, None))

    # Timing end and print elapsed time for this strategy
    strat_end_time = time.time()
    strat_elapsed = strat_end_time - strat_start_time
    print(f"→ {num_pits}-Pit Strategy Evaluation Time: {strat_elapsed:.2f} seconds")

    # Inserted block: Store best 4-pit strategy details if found
    if num_pits == 4 and best_plan:
        best_4_pit_strategy = {
            'pits': pits,
            'stints': stints,
            'modes': modes,
            'compounds': compounds,
            'time': best_time
        }


# After the multi-pit loop, print best 4-pit strategy summary if available
if 'best_4_pit_strategy' in locals():
    print("\n--- Best 4-Pit Strategy Summary ---")
    print(f"Pits at: {best_4_pit_strategy['pits']}")
    print(f"Stint Lengths: {best_4_pit_strategy['stints']}")
    print(f"Modes: {best_4_pit_strategy['modes']}")
    print(f"Tires: {best_4_pit_strategy['compounds']}")
    print(f"→ Total Race Time: {best_4_pit_strategy['time']:.2f}s")



import math
import time

# --- Step 2: Generalized Optimal Pit Strategy Calculation ---
optimal_function_indices = set()

func_mode_cache = {}
mode_factor = {'neutral': 1.0, 'aggressive': 1.1, 'durable': 0.9}

# --- Utility: Get stint mode function (caching for efficiency) ---

# === Helper functions for stint/mode/penalty logic ===
def compute_adjusted_stint_length(mode_str, stint_len):
    if '→' in mode_str and '@-' in mode_str:
        mode_sequence = mode_str.split('@-')[0].split('→')
        segment_lengths = list(map(int, mode_str.split('@-(')[1][:-1].split(',')))
        return sum(seg_len * mode_factor[m] for seg_len, m in zip(segment_lengths, mode_sequence))
    return stint_len * mode_factor[mode_str]

def get_mode_func(f_idx, mode):
    if (f_idx, mode) not in func_mode_cache:
        adjusted_expr = stint_mode_settings[f_idx][mode]
        func_mode_cache[(f_idx, mode)] = lambdify(u, adjusted_expr.subs(x, u), "numpy")
    return func_mode_cache[(f_idx, mode)]

def compute_penalty(f_idx, adjusted_len, soft_limit):
    if adjusted_len > soft_limit:
        over_soft = int(np.ceil(adjusted_len - soft_limit))
        over_soft = min(over_soft, len(penalty_cache))
        return penalty_growth_factors[f_idx] * penalty_cache[over_soft - 1]
    return 0

# --- Helper: Map stint function to its index for efficient lookup ---
func_index_map = {}
for idx, func in enumerate([stint1_func, stint2_func, stint3_func]):
    func_index_map[func] = idx

# --- Helper: Legality check for stints ---
def is_stint_legal(f_idx, mode, stint_len):
    hard_limit = stint_limits_array[f_idx][1]
    if '→' in mode and '@-' in mode:
        mode_sequence = mode.split('@-')[0].split('→')
        segment_lengths = list(map(int, mode.split('@-(')[1][:-1].split(',')))
        for m, seg_len in zip(mode_sequence, segment_lengths):
            adj_len = seg_len * mode_factor[m]
            if adj_len > hard_limit:
                return False
        total_adjusted = sum(seg_len * mode_factor[m] for m, seg_len in zip(mode_sequence, segment_lengths))
        return total_adjusted <= hard_limit
    else:
        adj_len = stint_len * mode_factor[mode]
        return adj_len <= hard_limit




# --- 3D Visualization of Pit Strategy Surface ---
from mpl_toolkits.mplot3d import Axes3D

def visualize_pit_strategy_surface():
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    pit1_range = range(min_lap_buffer, number_of_laps - min_lap_buffer)
    pit2_range = range(min_lap_buffer + min_lap_between_pits, number_of_laps - min_lap_buffer)

    X, Z, Y = [], [], []

    # Use only the stint functions found to be optimal in best_combo_info
    stint_funcs = [
        lambdify(u, best_combo_info[0].subs(x, u), "numpy"),
        lambdify(u, best_combo_info[1].subs(x, u), "numpy"),
        lambdify(u, best_combo_info[2].subs(x, u), "numpy"),
    ]

    for p1 in pit1_range:
        for p2 in pit2_range:
            if p2 - p1 >= min_lap_between_pits and p2 > p1:
                stint1_laps = np.arange(0, p1)
                stint2_laps = np.arange(0, p2 - p1)
                stint3_laps = np.arange(0, number_of_laps - p2)

                total_time = (
                    np.sum(stint_funcs[0](stint1_laps))
                    + average_pit_time
                    + np.sum(stint_funcs[1](stint2_laps))
                    + average_pit_time
                    + np.sum(stint_funcs[2](stint3_laps))
                )

                X.append(p1)
                Z.append(p2)
                Y.append(total_time)

    X = np.array(X)
    Z = np.array(Z)
    Y = np.array(Y)

    ax.scatter(X, Z, Y, c=Y, cmap='viridis', marker='o')

    # Highlight optimal
    opt_p1, opt_p2 = best_combo_info[3], best_combo_info[4]
    opt_time = best_combo_info[-1]
    ax.scatter(opt_p1, opt_p2, opt_time, c='red', s=100, label='Optimal Pit Stops')
    ax.text(opt_p1, opt_p2, opt_time, f"({opt_p1}, {opt_p2}, {opt_time:.1f})",
            color='red', fontsize=10, fontweight='bold', ha='left', va='bottom')

    # Determine nearest axis origin for each coordinate
    x_target = 0 if opt_p1 < number_of_laps / 2 else number_of_laps
    y_target = 0 if opt_p2 < number_of_laps / 2 else number_of_laps
    z_target = 0  # z-axis always extends downward

 



    ax.set_xlabel("Pit 1 Lap")
    ax.set_ylabel("Pit 2 Lap")
    ax.set_zlabel("Total Race Time (s)")
    ax.set_title("Race Time by Pit Stop Strategy")
    ax.legend()
    plt.tight_layout()
    plt.show()


# --- Plot Lap Times with Optimal Pit Strategy ---





import matplotlib.pyplot as plt
import numpy as np

def plot_all_optimal_strategies(results):
    """
    Plot lap times for all optimal strategies found in results.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Lap Times for All Optimal Pit Strategies")
    n_plots = 0
    for idx, res in enumerate(results):
        if res[1] is None:
            continue
        n_pits, plan = res
        func_combo = plan[:n_pits+1]
        pits = plan[n_pits+1]
        stints = plan[n_pits+2]
        modes = plan[n_pits+3]
        total_time = plan[n_pits+4]
        compounds = []
        for func in func_combo:
            for label, val in tire_exprs.items():
                if val == func:
                    compounds.append(label)
                    break
        lap_times_full = []
        lap_pointer = 0
        for i, (func, stint_len, mode) in enumerate(zip(func_combo, stints, modes)):
            f_idx = func_index_map[func]
            f = get_mode_func(f_idx, mode)
            laps = np.arange(stint_len)
            penalty = fatigue_penalty(laps, stint_limits_array[f_idx][0], penalty_growth_factors[f_idx])
            lap_times = np.array(f(laps), dtype=np.float64) + penalty
            # Inserted: Add pit stop time to first lap after a pit stop
            if i > 0:
                lap_times[0] += average_pit_time
            lap_times_full.extend(lap_times)
            lap_pointer += stint_len
        ax = axes.flat[n_plots]
        ax.plot(np.arange(1, len(lap_times_full)+1), lap_times_full, label=f"{n_pits} Pit(s)")
        ax.set_title(f"{n_pits}-Pit: Pits@{pits}\nTires: {compounds}\nModes: {modes}\nTotal: {total_time:.1f}s")
        ax.set_xlabel("Lap")
        ax.set_ylabel("Lap Time (s)")
        ax.legend()
        n_plots += 1
        if n_plots >= 4:
            break
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()





# --- Plot all optimal strategies in 2x2 grid ---
def plot_all_optimal_strategies(results):
    import matplotlib.pyplot as plt
    import numpy as np

    strategies = [r for r in results if r[1] is not None]
    n = len(strategies)
    if n == 0:
        print("No valid strategies to plot.")
        return
    cols = 2
    rows = (n + 1) // cols

    fig, axs = plt.subplots(rows, cols, figsize=(16, 4 * rows))
    axs = axs.flatten()

    # Inserted: Add driver name or AVERAGE to suptitle
    fig_driver = driver_filter.upper() if driver_filter else "AVERAGE"
    fig.suptitle(f"Lap Times for All Optimal Pit Strategies ({fig_driver})")

    for idx, (num_pits, strategy) in enumerate(strategies):
        *func_combo, pits, stints, modes, total_time = strategy
        ax1 = axs[idx]
        ax2 = ax1.twinx()

        cumulative_times = []
        lap_labels = []
        current_lap = 0
        lap_times_full = []
        # Get compounds for labeling
        compounds = []
        for func in func_combo:
            for label, val in tire_exprs.items():
                if val == func:
                    compounds.append(label)
                    break
        for i, (func, stint_len, mode) in enumerate(zip(func_combo, stints, modes)):
            f_idx = func_index_map[func]
            f = get_mode_func(f_idx, mode)
            laps = np.arange(stint_len)
            penalty = fatigue_penalty(laps, stint_limits_array[f_idx][0], penalty_growth_factors[f_idx])
            lap_times = np.array([float(val) for val in f(laps)], dtype=float) + penalty
            # Inserted: Add pit stop time to first lap after a pit stop
            if i > 0:
                lap_times[0] += average_pit_time  # Add pit stop time to first lap after pit
            lap_range = np.arange(current_lap, current_lap + stint_len)
            cumulative = np.cumsum(lap_times) + (cumulative_times[-1] if cumulative_times else 0)
            ax1.plot(lap_range, lap_times, label=f'Stint {i+1} ({compounds[i].upper()}, Mode: {mode})')
            # ax2.plot(lap_range, cumulative, color='grey', alpha=0.5)  # REMOVE/COMMENT OUT per instructions
            cumulative_times.extend(cumulative.tolist())
            lap_times_full.extend(lap_times.tolist())
            current_lap += stint_len
        # After this block, store adjusted cumulative lap times as well
        lap_times_full.extend(lap_times.tolist())

        # Plot single full uniform cumulative line outside stint loop
        total_laps = number_of_laps
        if len(cumulative_times) > 0:
            cumulative_line = np.array(cumulative_times)
            ax2.plot(np.arange(len(cumulative_line)), cumulative_line, color='grey', linestyle='-', linewidth=2, label="Cumulative")
            ax2.fill_between(np.arange(len(cumulative_line)), cumulative_line, color='grey', alpha=0.3)

        for j, pit in enumerate(pits):
            ax1.axvline(pit, linestyle='--', color='tab:blue', label='Pit(s)' if j == 0 else None)

        ax1.set_title(f"{num_pits} Pit Stop(s) - Total Time: {total_time:.1f}s")
        ax1.set_xlabel("Lap")
        ax1.set_ylabel("Lap Time (s)")
        ax2.set_ylabel("Cumulative Time (s)")
        ax1.legend(loc='upper left')

    plt.tight_layout()
    plt.show()

# Call the plotting function at the end
plot_all_optimal_strategies(results)

# --- Save precomputed_time_cache to disk after all computations ---
# --- Timing: End timer and print elapsed time before saving cache ---
end_time = time.time()
elapsed_time = end_time - start_time
print(f"\n=== Simulation Completed in {elapsed_time:.2f} seconds ===")
with open('precomputed_time_cache.pkl', 'wb') as f:
    pickle.dump(precomputed_time_cache, f)
