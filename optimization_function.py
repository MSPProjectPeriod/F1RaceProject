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

# --- Load stint functions from CSV ---
data_file = "csv_files/csv_results_0_2024_Bahrain_Grand_Prix_Grand_Prix.csv"
df = pd.read_csv(data_file)

number_of_laps = int(df['Total Laps'].iloc[0])  # Override from CSV

# Extract average equations by compound
compound_map = {'Soft': None, 'Medium': None, 'Hard': None}
for compound in compound_map.keys():
    match = df[
        df['driver'].str.upper().str.startswith("AVERAGE_EQUATION", na=False) &
        (df['tiretype'].str.upper() == compound.upper())
    ]
    if not match.empty:
        compound_map[compound] = match.iloc[0]['besttrendline']

# Map compound functions
tire_exprs = {}
if compound_map['Soft']:
    tire_exprs['soft'] = sympify(compound_map['Soft'].replace("^", "**"))
if compound_map['Medium']:
    tire_exprs['medium'] = sympify(compound_map['Medium'].replace("^", "**"))
if compound_map['Hard']:
    tire_exprs['hard'] = sympify(compound_map['Hard'].replace("^", "**"))

# Assign only available stint functions in preferred order: Soft, Medium, Hard
stint_exprs = []
if 'soft' in tire_exprs:
    stint_exprs.append(tire_exprs['soft'])
if 'medium' in tire_exprs:
    stint_exprs.append(tire_exprs['medium'])
if 'hard' in tire_exprs:
    stint_exprs.append(tire_exprs['hard'])

if not stint_exprs:
    raise ValueError("No valid tire performance functions found in the file.")

# Fallback: Assign symbolic names
stint1_func, stint2_func, stint3_func = (stint_exprs + [stint_exprs[-1]] * 3)[:3]

# Ensure at least two valid stint functions
if len(stint_exprs) < 2:
    raise ValueError("At least two valid tire performance functions are required. Only found: " + ', '.join(tire_exprs.keys()))


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
            match = df[
                df['driver'].str.upper().str.startswith("AVERAGE_EQUATION", na=False) &
                (df['tiretype'].str.lower() == label)
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



# Now print only legal pit stop locations where both stints are within durable-mode hard limits:
print("\n--- One-Pit Stop Strategy Candidates (Filtered by Durable Limits) ---")
print(f"Listing only legal pit stop locations where both stints are within durable-mode hard limits:")

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
        func_mode_cache[(f_idx, mode)] = lambdify(u, adjusted_expr.subs(x, u), "numpy")
    return func_mode_cache[(f_idx, mode)]

 # --- Precompute penalty cache ---
max_penalty_len = number_of_laps
penalty_cache = np.cumsum(np.exp(np.arange(1, max_penalty_len + 1) / 5) - 1)

# Fatigue penalty function (uses penalty_cache)
def fatigue_penalty(t_array, soft_limit, factor):
    return np.zeros_like(t_array)

best_strategy = None
best_total_time = float('inf')
for pit in range(min_lap_buffer + 1, number_of_laps - min_lap_buffer + 1):
    stint1_len = pit
    stint2_len = number_of_laps - pit

    # Use the first two stint functions and determine the corresponding compound labels
    f1_idx, f2_idx = 0, 1
    func1 = stint_exprs[f1_idx]
    func2 = stint_exprs[f2_idx]

    compound1 = None
    compound2 = None
    for label, func in tire_exprs.items():
        if func == func1:
            compound1 = label
        if func == func2:
            compound2 = label

    if not compound1 or not compound2:
        continue  # Skip if compound mapping is missing

    max1 = tire_max_limits[compound1]['durable']['hard_limit']
    max2 = tire_max_limits[compound2]['durable']['hard_limit']

    if stint1_len <= max1 and stint2_len <= max2:
        print(f"  P1: {pit:<2} | S1: {stint1_len:<2} | S2: {stint2_len:<2}")

        valid_modes = []
        for mode in ['durable', 'neutral', 'aggressive']:
            max1_mode = tire_max_limits[compound1][mode]['hard_limit']
            max2_mode = tire_max_limits[compound2][mode]['hard_limit']
            if stint1_len <= max1_mode and stint2_len <= max2_mode:
                valid_modes.append(mode.capitalize())

        if valid_modes:
            print(f"    ↳ Valid in mode(s): {', '.join(valid_modes)}")
            # For each valid mode combination, calculate total race time and print strategy details
            from itertools import product
            for mode_combo in product(valid_modes, repeat=2):
                mode1, mode2 = mode_combo
                f1_idx = func_index_map[func1]
                f2_idx = func_index_map[func2]
                f1 = get_mode_func(f1_idx, mode1.lower())
                f2 = get_mode_func(f2_idx, mode2.lower())
                laps1 = np.arange(stint1_len)
                laps2 = np.arange(stint2_len)
                penalty1 = fatigue_penalty(laps1, stint_limits_array[f1_idx][0], penalty_growth_factors[f1_idx])
                penalty2 = fatigue_penalty(laps2, stint_limits_array[f2_idx][0], penalty_growth_factors[f2_idx])
                time1 = np.sum(f1(laps1) + penalty1)
                time2 = np.sum(f2(laps2) + penalty2)
                total_time = time1 + time2 + average_pit_time
                print(f"    → Modes: S1={mode1}, S2={mode2}, Total Race Time: {total_time:.2f}s")
                if total_time < best_total_time:
                    best_total_time = total_time
                    best_strategy = {
                        'P1': pit,
                        'S1': stint1_len,
                        'S2': stint2_len,
                        'modes': (mode1, mode2),
                        'time': total_time
                    }

# After the loop over all pit stops, print the best one-stop strategy if found

if best_strategy:
    print("\n--- Best One-Stop Strategy ---")
    print(f"P1: {best_strategy['P1']} Modes: S1={best_strategy['modes'][0]}, S2={best_strategy['modes'][1]} | Total Time: {best_strategy['time']:.2f}s")


# === Multi-Pit Stop Strategy Candidates ===
from itertools import combinations, product

print("\n--- Multi-Pit Stop Strategy Candidates ---")

from itertools import product as iter_product
from itertools import combinations_with_replacement

# Store results for each multi-pit strategy
results = []

def plot_strategy_lap_profile(strategy):
    import matplotlib.pyplot as plt

    *func_combo, pit_laps, stint_lengths, modes, total_time = strategy

    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Prepare lap-by-lap times and cumulative time using stint segments
    cumulative_times = []
    lap_labels = []
    compound_labels = []

    # Draw each stint as an independent segment
    cumulative_times = []
    current_lap = 0
    ax2 = ax1.twinx()
    # Compute compound labels for each stint
    compound_labels_stint = []
    for func, mode in zip(func_combo, modes):
        label = None
        for name, expr in tire_exprs.items():
            if expr == func:
                label = f"{name.capitalize()} ({mode})"
                break
        compound_labels_stint.append(label)

    for i, (stint_len, label) in enumerate(zip(stint_lengths, compound_labels_stint)):
        lap_range = np.arange(current_lap, current_lap + stint_len)
        f_idx = func_index_map[func_combo[i]]
        f = get_mode_func(f_idx, modes[i])
        laps = np.arange(stint_len)
        penalties = fatigue_penalty(laps, stint_limits_array[f_idx][0], penalty_growth_factors[f_idx])
        lap_times_segment = f(laps) + penalties
        # Insert pit stop time at the start of each stint after the first (visual offset)
        if i > 0:
            lap_times_segment[0] += average_pit_time  # simulate pit time in lap time
        # Plot the segment
        ax1.plot(lap_range, lap_times_segment, label=f'Stint {i+1}: {label}', linewidth=2)
        # Cumulative time, with pit stop time for non-first stints
        if cumulative_times:
            cumulative_segment = np.cumsum(lap_times_segment) + cumulative_times[-1] + average_pit_time
        else:
            cumulative_segment = np.cumsum(lap_times_segment)
        # Instead of plotting each segment, collect for a single line after the loop
        lap_labels.extend(lap_range)
        cumulative_times.extend(cumulative_segment)
        compound_labels.extend([label] * stint_len)
        current_lap += stint_len

    lap_labels = np.array(lap_labels)
    # After collecting all lap_labels and cumulative_times, plot a single cumulative line
    ax2.plot(lap_labels, cumulative_times, color='grey')
    ax2.fill_between(lap_labels, cumulative_times, color='grey', alpha=0.2)
    # Plot settings
    ax1.set_xlabel("Lap")
    ax1.set_ylabel("Time per Lap (s)", color='tab:blue')
    ax1.set_ylim(90, 130)
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    ax2.set_ylabel("Cumulative Time (s)", color='grey')
    ax2.tick_params(axis='y', labelcolor='grey')

    # Add pit stop lines
    for pit in pit_laps:
        ax1.axvline(pit, color='red', linestyle='--', linewidth=1)

    # Annotate stint functions
    prev = 0
    for i, stint_len in enumerate(stint_lengths):
        mid = prev + stint_len // 2
        if compound_labels:
            ax1.text(mid, max(ax1.get_ylim()) * 0.95, compound_labels[prev], rotation=0,
                     ha='center', va='bottom', fontsize=9, color='black')
        prev += stint_len

    plt.title("Race Lap Profile")
    ax1.legend(loc='upper left', fontsize='small', title='Stint Legend')
    plt.tight_layout()
    plt.show()


for num_pits in [2, 3, 4]:
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
        # Plot lap profile for this strategy
        plot_strategy_lap_profile(formatted_plan)
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

# --- Plot all optimal strategies in 2x2 grid ---
def plot_all_strategies(results):
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    axs = axs.flatten()

    for idx, (num_pits, strategy) in enumerate(results):
        ax = axs[idx]
        ax.set_title(f"{num_pits}-Pit Strategy")
        ax.set_xlabel("Stint")
        ax.set_ylabel("Time per Stint (s)")

        if strategy is None:
            ax.text(0.5, 0.5, "No valid strategy found", ha='center', va='center', transform=ax.transAxes)
            continue

        *func_combo, pits, stints, modes, total_time = strategy
        compound_labels = []
        stint_times = []

        for i, (func, stint_len, mode) in enumerate(zip(func_combo, stints, modes)):
            for label, expr in tire_exprs.items():
                if expr == func:
                    compound_labels.append(f"{label.capitalize()} ({mode})")
                    f_idx = func_index_map[func]
                    f = get_mode_func(f_idx, mode)
                    laps = np.arange(stint_len)
                    penalty = fatigue_penalty(laps, stint_limits_array[f_idx][0], penalty_growth_factors[f_idx])
                    stint_time = np.sum(f(laps) + penalty)
                    stint_times.append(stint_time)
                    break

        ax.bar(range(len(stint_times)), stint_times, tick_label=compound_labels)
        if stint_times:
            ax.set_ylim(0, max(stint_times) * 1.2)
        ax.text(0.5, 1.05, f"Total: {total_time:.2f}s", transform=ax.transAxes, ha='center', fontsize=10)

    plt.tight_layout()
    plt.show()

plot_all_strategies(results)




# --- Save precomputed_time_cache to disk after all computations ---
# --- Timing: End timer and print elapsed time before saving cache ---
end_time = time.time()
elapsed_time = end_time - start_time
print(f"\n=== Simulation Completed in {elapsed_time:.2f} seconds ===")
with open('precomputed_time_cache.pkl', 'wb') as f:
    pickle.dump(precomputed_time_cache, f)
