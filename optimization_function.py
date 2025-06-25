# === F1 Race Strategy Optimizer ===
# This script optimizes pit stop strategies and lap times for F1 races based on tire data.
#
# === Key Variables ===
# number_of_laps: Total laps in the race, set based on driver data or default.
# average_pit_time: Average time (seconds) added per pit stop.
# min_lap_buffer: Laps at race start/end where pit stops are disallowed.
# min_lap_between_pits: Minimum number of laps between pit stops.
# degradation_factors: Lap progression and time scaling for each driving mode.
# tire_exprs: Parsed symbolic equations for each tire compound.
# stint_exprs: List of tire performance functions for each stint.
# compound_limits: Soft/hard stint length limits for each compound.
# tire_max_limits: Precomputed soft/hard stint limits for each compound and mode.
# allowed_mode_cache: Maps (compound, stint_len) to allowed modes.
# stint_mode_settings: Maps (stint function index, mode) to adjusted symbolic function.
# func_mode_cache: Caches numeric lap time functions for (stint func index, mode).
# precomputed_time_cache: Caches computed stint times to avoid recalculation.
# results: List of (num_pits, best_strategy) tuples storing optimal strategies.
# strategy_data_array: List of all evaluated strategy metadata for analysis.
# best_combo_info: Stores best pit combo info (if visualization is needed).
# driver_filter: Selected driver (or 'AVERAGE') for data filtering.
# selected_file: Path to the race data file used.
# DEFAULT_CSV_FILE: Default race data file path.
# CSV_FILES_DIR: Directory containing race data CSVs.

# --- Timing: Start timer at script start ---

# === All Imports at the Top ===
import os
import pickle
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sympy import symbols, sympify, integrate, lambdify, diff
from sympy.parsing.sympy_parser import parse_expr
from itertools import combinations, product, combinations_with_replacement
from mpl_toolkits.mplot3d import Axes3D
from collections import defaultdict
import re
import math
import time

# === File Paths (edit these if running on a different computer) ===
CSV_FILES_DIR = '/Users/foml/coding/MSP/period 6/F1RaceProject/csv_files'
DEFAULT_CSV_FILE = '/Users/foml/coding/MSP/period 6/F1RaceProject/csv_files/csv_results_0_2024_Bahrain_Grand_Prix_Grand_Prix.csv'
PRECOMPUTED_CACHE_FILE = 'precomputed_time_cache.pkl'

# --- Start timer for total runtime measurement ---
start_time = time.time()


# --- Precomputed time cache (global) ---
precomputed_time_cache = {}
# Try to load cache from disk if exists
if os.path.exists(PRECOMPUTED_CACHE_FILE):
    with open(PRECOMPUTED_CACHE_FILE, 'rb') as f:
        precomputed_time_cache = pickle.load(f)

# --- Global Configuration Variables ---
number_of_laps = 100             # Total number of laps in the race (max lap limit)
average_pit_time = 30.0          # Default average pit stop time (seconds)
min_lap_buffer = 5               # Laps to avoid at start/end before a pit may occur
min_lap_between_pits = 5         # Minimum laps between two pit stops
mode_alpha = 0.2                 # Controls how aggressive/durable behavior affects lap time
best_combo_info = None           # Stores optimal pit stop info for later visualization



# Stint limits for each function: [soft_limit, hard_limit]
stint_limits_array = [
    [15, 20],   # Stint 1 [soft, hard]
    [15, 25],   # Stint 2 [soft, hard]
    [20, 30],   # Stint 3 [soft, hard]
]

# Mode impact on lap time and tire life
# Each mode adjusts lap time and affects how long tires last.
mode_degradation_factors = {
    'aggressive': {'lap_time_multiplier': 0.9, 'tire_life_multiplier': 0.9},
    'neutral': {'lap_time_multiplier': 1.0, 'tire_life_multiplier': 1.0},
    'durable': {'lap_time_multiplier': 1.1, 'tire_life_multiplier': 1.1}
}





# Symbolic variables for lap calculations
x = symbols('x')                 # Primary lap counter for each stint
u = symbols('u')                 # Reset lap counter after a pit stop

def get_race_files_info(folder=CSV_FILES_DIR):
    """
    Scan the CSV_FILES_DIR for race result files and extract season/race info.
    """
    race_files = []
    for filename in os.listdir(folder): 
        if "csv_results" in filename and filename.endswith(".csv"):
            path = os.path.join(folder, filename)
            try:
                df_sample = pd.read_csv(path, nrows=5)
                season = None
                race = None
                # Attempt to find season and race name columns (case insensitive)
                for col in df_sample.columns:
                    if col.lower() == "season":
                        season = df_sample[col].iloc[0]
                    if col.lower() in ["race", "race_name", "race title"]:
                        race = df_sample[col].iloc[0]
                # Fallback: try to parse from filename if missing
                if season is None:
                    # Try to find 4-digit year in filename
                    match = re.search(r"\b(20\d{2})\b", filename)
                    if match:
                        season = int(match.group(1))
                if race is None:
                    # Use filename part as fallback
                    race = filename.replace(".csv", "").replace("_", " ")
                race_files.append({"filename": filename, "season": season, "race": race, "path": path})
            except Exception as e:
                print(f"Failed to read {filename}: {e}")
    # Sort by season then race name
    race_files.sort(key=lambda x: (x["season"] if x["season"] is not None else 0, x["race"]))
    return race_files

def select_race_file():
    """
    Prompt user to select a race file from available CSVs.
    """
    races = get_race_files_info()
    if not races:
        print("No race data files found in csv_files/. Using default.")
        return None
    print("Available race data files:")
    for i, race in enumerate(races):
        print(f"{i+1}. Season {race['season']} - {race['race']} ({race['filename']})")
    choice = input("Select a race by number (or leave blank for default): ")


    if choice == "":
        return None
    try:
        idx = int(choice) - 1
        if 0 <= idx < len(races):
            return races[idx]["path"]
    except Exception:
        pass
    print("Invalid selection. Using default race data.")
    return None

# --- Load race data file and filter by driver ---
selected_file = select_race_file()
if selected_file:
    print(f"Loading race data from: {selected_file}")
    df = pd.read_csv(selected_file)
else:
    default_file = DEFAULT_CSV_FILE
    if os.path.exists(default_file):
        print(f"Loading default race data from: {default_file}")
        df = pd.read_csv(default_file)
    else:
        # Fallback: Use first valid CSV in the folder
        race_files = get_race_files_info()
        if race_files:
            fallback_file = race_files[0]['path']
            print(f"⚠️ Default file not found. Loading fallback race data from: {fallback_file}")
            df = pd.read_csv(fallback_file)
        else:
            raise FileNotFoundError("❌ No default or fallback race CSV found in the directory.")

# Prompt user for driver filter or use average data
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

# --- Extract average equations and std for each tire compound ---
compound_map = {'Soft': None, 'Medium': None, 'Hard': None}
compound_std_map = {}
for compound in compound_map.keys():
    match = filtered_df[
        (filtered_df['tiretype'].str.upper() == compound.upper())
    ]
    if not match.empty:
        compound_map[compound] = match.iloc[0]['besttrendline']
        std_val = match.iloc[0].get('residual_std', match.iloc[0].get('std_err', None))
        if pd.isnull(std_val):
            std_val = 0.0
            print(f"⚠️ Missing std for {compound}. Setting std to 0.0.")
        else:
            print(f"✅ Extracted std for {compound}: {std_val}")
        compound_std_map[compound.lower()] = std_val
    else:
        compound_std_map[compound.lower()] = 0.0
        print(f"⚠️ No data for {compound}. Setting std to 0.0.")

# Warn if any compound formula is missing
for compound, formula in compound_map.items():
    if formula is None:
        print(f"⚠️ Missing formula for: {compound}")

# --- Parse tire performance equations ---
def parse_equation_string(expr_str):
    """
    Parse a string equation into a sympy expression.
    """
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

# Assign stint functions in preferred order: Soft, Medium, Hard
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

# Fallback: Assign symbolic names for up to 3 stints
stint1_func, stint2_func, stint3_func = (stint_exprs + [stint_exprs[-1]] * 3)[:3]


# --- Degradation factors for each driving mode ---
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


print(f"\nTotal Number of Laps in Race: {number_of_laps}")

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











# --- Stint mode settings (adjusted equations for each mode) ---
stint_base_exprs = [stint1_func, stint2_func, stint3_func]
mode_order = ['aggressive', 'neutral', 'durable']  # Mode order for reference

# Precompute adjusted stint equations for each mode
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

# --- Function cache for stint/mode lambdified functions ---
if 'func_mode_cache' not in globals():
    func_mode_cache = {}
def get_mode_func(f_idx, mode):
    """
    Return a lambdified function for the given stint index and mode.
    """
    if (f_idx, mode) not in func_mode_cache:
        adjusted_expr = stint_mode_settings[f_idx][mode]
        expr_num = adjusted_expr.subs({x: u})  # ensure all symbolic variables are replaced
        func_mode_cache[(f_idx, mode)] = lambdify(u, expr_num, modules="numpy")
    return func_mode_cache[(f_idx, mode)]

# --- Penalty cache for stint length penalties (if used) ---
max_penalty_len = number_of_laps
penalty_cache = np.cumsum(np.exp(np.arange(1, max_penalty_len + 1) / 5) - 1)




# --- Results and strategy storage ---
results = []
simulation_times_array = []
strategy_data_array = []


# === Multi-Pit Stop Strategy Candidates ===
print("\n--- Multi-Pit Stop Strategy Candidates ---")

# Loop over possible pit stop counts (1-4)
for num_pits in [1, 2, 3, 4]:
    num_stints = num_pits + 1
    possible_func_combos = []
    seen_compound_sets = set()

    # Generate all possible combinations of stint functions for the number of stints
    for func_combo in product(stint_exprs, repeat=num_stints):
        compounds_used = []
        for func in func_combo:
            for label, expr in tire_exprs.items():
                if expr == func:
                    compounds_used.append(label)
        # Only consider combos using at least 2 different compounds
        if len(set(compounds_used)) >= 2:
            key = tuple(sorted(compounds_used))
            if key not in seen_compound_sets:
                seen_compound_sets.add(key)
                possible_func_combos.append(func_combo)

    pit_positions = range(min_lap_buffer, number_of_laps - min_lap_buffer)
    seen_keys = set()  # Deduplicate strategies
    for func_combo in possible_func_combos:
        for pit_combo in combinations(pit_positions, num_pits):
            # Enforce minimum laps between pit stops
            if any(pit_combo[i+1] - pit_combo[i] < min_lap_between_pits for i in range(len(pit_combo)-1)):
                continue
            # Calculate stint lengths from pit positions
            stints = [pit_combo[0]] + [pit_combo[i+1] - pit_combo[i] for i in range(len(pit_combo)-1)] + [number_of_laps - pit_combo[-1]]
            if any(s <= 0 for s in stints):
                continue
            # Map stint functions to compound labels
            compounds = []
            for func in func_combo:
                for label, val in tire_exprs.items():
                    if val == func:
                        compounds.append(label)
                        break
            # --- Begin new mode trial logic ---
            from itertools import combinations
            n_stints = len(stints)
            valid = False

            def check_valid(modes):
                """
                Check if a mode assignment is valid for the given stints and compounds.
                """
                hard_limit_sum = sum(
                    tire_max_limits[compound][mode]['hard_limit']
                    for stint_len, compound, mode in zip(stints, compounds, modes)
                )
                if hard_limit_sum < number_of_laps:
                    return False
                if all(
                    stint_len <= tire_max_limits[compound][mode]['hard_limit']
                    for stint_len, compound, mode in zip(stints, compounds, modes)
                ):
                    print(f"✅ Viable: Pits: {pit_combo}, Stints: {stints}, Compounds: {compounds}, Modes: {modes}, Tire Max Life Sum: {hard_limit_sum}")
                    return True
                return False

            # Try all-aggressive, then all-neutral, all-durable, then mixed
            if check_valid(['aggressive'] * n_stints):
                modes = ['aggressive'] * n_stints
                valid = True
            else:
                mode_tried = False
                for base in [['neutral'] * n_stints, ['durable'] * n_stints]:
                    if check_valid(base):
                        modes = base
                        valid = True
                        mode_tried = True
                        break
                if not valid:
                    for base in [['neutral'] * n_stints, ['durable'] * n_stints]:
                        for r in range(1, n_stints + 1):
                            for indices in combinations(range(n_stints), r):
                                trial_modes = base.copy()
                                for i in indices:
                                    trial_modes[i] = 'aggressive'
                                if check_valid(trial_modes):
                                    modes = trial_modes
                                    valid = True
                                    mode_tried = True
                                    break
                            if valid:
                                break
                        if valid:
                            break
                # If nothing valid found, default to all durable
                if not valid:
                    modes = ['durable'] * n_stints
                    if check_valid(modes):
                        valid = True
            # --- End replaced mode trial logic ---
            if valid:
                # Compute total variance for this strategy
                total_variance = 0.0
                for stint_len, compound in zip(stints, compounds):
                    compound_sigma = compound_std_map.get(compound, 0.0)
                    total_variance += stint_len * (compound_sigma ** 2)
                std_dev = math.sqrt(total_variance)
                key = (
                    len(pit_combo),
                    compounds[0],
                    tuple(sorted(zip(stints[1:], compounds[1:], modes[1:])))
                )
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                # Store strategy metadata for later analysis
                strategy_data_array.append({
                    "num_pits": len(pit_combo),
                    "pit_combo": pit_combo,
                    "stints": stints,
                    "compounds": compounds,
                    "modes": modes,
                    "std_dev": std_dev
                })

# --- Summarize best strategy per pit count ---
best_per_pits = {}
strategies_by_pits = defaultdict(list)
for strat in strategy_data_array:
    strategies_by_pits[strat["num_pits"]].append(strat)

for pits, strats in strategies_by_pits.items():
    if not strats:
        continue
    # Only consider strategies that can cover the race distance
    valid_strats = [
        s for s in strats
        if sum(
            tire_max_limits[compound][mode]['hard_limit']
            for compound, mode in zip(s["compounds"], s["modes"])
        ) >= number_of_laps
    ]
    if not valid_strats:
        continue
    # Pick the strategy with the lowest total tire max life
    best = min(
        valid_strats,
        key=lambda s: sum(
            tire_max_limits[compound][mode]['hard_limit']
            for compound, mode in zip(s["compounds"], s["modes"])
        )
    )
    best_per_pits[pits] = best
    # Do NOT append to results here; only append fully formed dicts with total_time in print_initial_optimal_strategies



# --- Step 2: Generalized Optimal Pit Strategy Calculation ---
optimal_function_indices = set()

# --- Helper: Legality check for stints ---
def is_stint_legal(f_idx, mode, stint_len):
    """
    Check if a stint is legal for the given function index, mode, and stint length.
    """
    hard_limit = stint_limits_array[f_idx][1]
    mode_factor = {'neutral': 1.0, 'aggressive': 0.9, 'durable': 1.1}
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
def visualize_pit_strategy_surface():
    """
    Plot a 3D surface of total race time as a function of pit stop laps.
    """
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    # This dynamic version supports arbitrary pit stop counts; update combinations and plotting as needed.
    from itertools import combinations
    pit_range = range(min_lap_buffer, number_of_laps - min_lap_buffer)
    X, Z, Y = [], [], []
    # Use only the stint functions found to be optimal in best_combo_info
    stint_funcs = [
        lambdify(u, best_combo_info[0].subs(x, u), "numpy"),
        lambdify(u, best_combo_info[1].subs(x, u), "numpy"),
        lambdify(u, best_combo_info[2].subs(x, u), "numpy"),
    ]
    # For now, assume 2 pit stops for plotting (3 stints)
    num_pits = 2
    for pit_combo in combinations(pit_range, num_pits):
        if any(pit_combo[i+1] - pit_combo[i] < min_lap_between_pits for i in range(len(pit_combo)-1)):
            continue
        stints = [pit_combo[0]] + [pit_combo[i+1] - pit_combo[i] for i in range(len(pit_combo)-1)] + [number_of_laps - pit_combo[-1]]
        if any(s <= 0 for s in stints):
            continue
        total_time = 0
        for idx, stint_len in enumerate(stints):
            laps = np.arange(0, stint_len)
            total_time += np.sum(stint_funcs[idx](laps))
            if idx < len(stints) - 1:
                total_time += average_pit_time
        X.append(pit_combo[0])
        Z.append(pit_combo[1])
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


# --- Utility: Get plot save prefix with folder ---
def get_plot_save_prefix():
    """
    Generate a filename prefix for saving plots based on race/driver info.
    """
    plots_dir = "plots"
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)
    if selected_file:
        fname = os.path.basename(selected_file)
    else:
        fname = os.path.basename(DEFAULT_CSV_FILE)
    # Extract year and race name from filename
    match = re.search(r'(\d{4})_(.*?)_Grand_Prix', fname)
    if match:
        year = match.group(1)
        race = match.group(2).replace("_", " ")
    else:
        year = "unknown"
        race = "unknown"
    # Get season from DataFrame if possible
    try:
        season = str(int(df['season'].iloc[0]))
    except:
        season = "unknown"
    driver = driver_filter.upper() if driver_filter else "AVERAGE"
    # Clean names for filenames
    race_clean = race.replace(" ", "_")
    driver_clean = driver.replace(" ", "_")
    return f"{plots_dir}/{season}_{year}_{race_clean}_{driver_clean}"

def plot_all_optimal_strategies(results):
    """
    Plot lap times for all optimal strategies (1-4 pit stops) in subplots.
    """

    strategies_dict = {p: s for p, s in results if s is not None}
    fig, axs = plt.subplots(2, 2, figsize=(14, 8))
    axs = axs.flatten()

    for i, pit_count in enumerate([1, 2, 3, 4]):
        ax = axs[i]
        strategy = strategies_dict.get(pit_count)
        if strategy is None:
            ax.set_title(f"{pit_count} Pit Stop(s) - No valid strategy")
            ax.axis('off')
            continue

        total_laps = sum(strategy['stints'])
        times = np.zeros(total_laps, dtype=float)

        lap_offset = 0
        pit_stops = np.cumsum(strategy['stints'])[:-1]

        num_stints = len(strategy['stints'])
        for stint_idx, (stint_len, compound, mode) in enumerate(zip(strategy['stints'], strategy['compounds'], strategy['modes'])):
            f_idx = func_index_map[tire_exprs[compound]]
            mode_func = get_mode_func(f_idx, mode)
            # Determine the correct lap range for each stint
            # For all but the last stint, include the pit stop lap (so add +1 to stint_len)
            if stint_idx < num_stints - 1:
                laps = np.arange(0, stint_len + 1)
            else:
                laps = np.arange(0, stint_len)
            stint_times = mode_func(laps)
            # --- Inserted: Apply ±10% lap time adjustment by mode ---
            mode_time_multiplier = {'neutral': 1.0, 'aggressive': 0.9, 'durable': 1.1}
            stint_times = stint_times * mode_time_multiplier[mode]

            # Lap numbers in the global race for this stint
            if stint_idx < num_stints - 1:
                lap_numbers = np.arange(lap_offset, lap_offset + stint_len + 1)
            else:
                lap_numbers = np.arange(lap_offset, lap_offset + stint_len)

            if stint_idx > 0:
                stint_times[0] += average_pit_time

            ax.plot(
                lap_numbers + 1,  # +1 so lap 0 is lap 1
                stint_times,
                label=f"Stint {stint_idx+1} ({compound.upper()}, Mode: {mode})"
            )

            std_val = compound_std_map.get(compound, 0.0)
            ax.plot(
                lap_numbers + 1,
                stint_times + 3 * std_val,
                linestyle='--', color='orange',
                label='Lap +3σ' if stint_idx == 0 else ""
            )
            ax.plot(
                lap_numbers + 1,
                stint_times - 3 * std_val,
                linestyle='--', color='orange',
                label='Lap -3σ' if stint_idx == 0 else ""
            )

            # Assign the calculated stint times to the global times array, but only for valid indices
            # For all but the last stint, the last lap overlaps with the next stint's first lap (pit stop lap),
            # so only fill up to the last lap (not including the extra pit stop lap) in the times array.
            if stint_idx < num_stints - 1:
                # Only fill the first stint_len laps (exclude the extra pit stop lap from times array)
                times[lap_numbers[:stint_len]] = stint_times[:stint_len]
                lap_offset += stint_len
            else:
                # Last stint: fill all
                times[lap_numbers] = stint_times
                lap_offset += stint_len

        for pit_lap in pit_stops:
            ax.axvline(x=pit_lap + 1, color='red', linestyle='--', linewidth=1.5, label='Pit Stop' if pit_lap == pit_stops[0] else "")

        # === Add vertical lines and scatter points at pit stop boundaries for both adjacent stints ===
        for pit_lap in pit_stops:
            ax.axvline(x=pit_lap + 1, color='red', linestyle='--', linewidth=1.5, label='Pit Stop' if pit_lap == pit_stops[0] else "")

        # === Plot cumulative time on right Y axis ===
        # Inserted block here
        # Plot cumulative time on right Y axis
        cum_ax = ax.twinx()
        cumulative_time = np.cumsum(times)
        cum_ax.plot(np.arange(1, total_laps + 1), cumulative_time, color='grey', alpha=0.7, label='Cumulative Time')
        cum_ax.fill_between(np.arange(1, total_laps + 1), 0, cumulative_time, color='grey', alpha=0.2)
        cum_ax.set_ylabel("Cumulative Time (s)")

        total_time = strategy.get('total_time', 0.0)
        std_dev = strategy.get('std_dev', 0.0)
        ax.set_title(f"{pit_count} Pit Stop(s) - Total Time: {total_time:.1f}s ± {3*std_dev:.1f}s")
        ax.set_xlabel("Lap")
        ax.set_ylabel("Lap Time (s)")
        # Dynamically set y-axis limits based on lap times
        y_min = min(times[times > 0])
        y_max = max(times)
        ax.set_ylim(y_min - 10, y_max + 10)
        ax.grid()

        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc='upper left')

    fig.suptitle(f"Lap Times for All Optimal Pit Strategies ({driver_filter.upper() if driver_filter else 'AVERAGE'})")
    plt.tight_layout()
    plt.show()

def plot_strategy_summary_with_std(results):
    """
    Plot total race time ± 3σ for each optimal strategy as points with error bars.
    """

    labels = []
    times = []
    std_devs = []

    for num_pits, strategy in results:
        if strategy is None:
            continue
        if 'total_time' not in strategy:
            print(f"⚠️ Skipping strategy for {num_pits} pit(s) — missing total_time")
            continue
        labels.append(f"{num_pits} Pit(s)")
        times.append(strategy['total_time'])
        std_devs.append(3 * strategy.get('std_dev', 0))  # Default to 0 if not present

    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(10, 6))
    fig_driver = driver_filter.upper() if driver_filter else "AVERAGE"
    fig.suptitle(f"Lap Times for All Optimal Pit Strategies ({fig_driver})")
    # Show only individual points with error bars (no connecting line)
    ax.errorbar(
        x, times, yerr=std_devs,
        fmt='o', capsize=5, elinewidth=2, markersize=6, color='black', linestyle='None'
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Total Race Time (s)")
    ax.set_title("Comparison of Optimal Strategies with ±3σ Error Bars")

    for i, (xi, yi) in enumerate(zip(x, times)):
        ax.annotate(f'{yi:.1f}s',
                    xy=(xi, yi),
                    xytext=(0, 6),
                    textcoords="offset points",
                    ha='center', va='bottom')

    plt.tight_layout()
    # Save the figure
    save_prefix = get_plot_save_prefix()
    fig.savefig(f"{save_prefix}_summary_stddev.png")
    plt.show()

def print_initial_optimal_strategies(strategy_data_array):
    """
    Print the initial optimal strategy for each pit stop count (durable mode results).
    """
    from collections import defaultdict
    strategies_by_pits = defaultdict(list)
    for strat in strategy_data_array:
        strategies_by_pits[strat["num_pits"]].append(strat)

    for pits, strats in strategies_by_pits.items():
        if not strats:
            continue
        valid_strats = [
            s for s in strats
            if sum(
                tire_max_limits[compound][mode]['hard_limit']
                for stint_len, compound, mode in zip(s["stints"], s["compounds"], s["modes"])
            ) >= number_of_laps
        ]
        if not valid_strats:
            continue
        best = min(
            valid_strats,
            key=lambda s: sum(
                tire_max_limits[compound][mode]['hard_limit']
                for stint_len, compound, mode in zip(s["stints"], s["compounds"], s["modes"])
            )
        )
        print(f"\n ------- Initial Optimal Strategy (Lowest Tire Max Life) for {pits} Pit Stop(s): -------")
        print(f"  Pit Stops: {best['pit_combo']}")
        print(f"  Stints: {best['stints']}")
        print(f"  Compounds: {best['compounds']}")
        print(f"  Modes: {best['modes']}")
        tire_life_sum = sum(
            tire_max_limits[compound]['durable']['hard_limit']
            for compound in best['compounds']
        )
        print(f"  Theoretical Max Tire Life: {tire_life_sum} laps")
        std_dev = best.get('std_dev', 0.0)
        total_variance = 0.0
        for stint_len, compound in zip(best['stints'], best['compounds']):
            compound_sigma = compound_std_map.get(compound, 0.0)
            total_variance += stint_len * (compound_sigma ** 2)
        std_dev = math.sqrt(total_variance)
        total_time = len(best["pit_combo"]) * average_pit_time
        mode_time_multiplier = {'neutral': 1.0, 'aggressive': 0.9, 'durable': 1.1}
        for stint_len, compound, mode in zip(best["stints"], best["compounds"], best["modes"]):
            f_idx = func_index_map[tire_exprs[compound]]
            mode_func = get_mode_func(f_idx, mode)
            laps = np.arange(0, stint_len)
            stint_times = mode_func(laps) * mode_time_multiplier[mode]
            total_time += np.sum(stint_times)
        print(f"  Total Time: {total_time:.2f} s ± {3 * std_dev:.2f}s (3σ)")
        # --- Append to results for plotting functions (only here, with total_time) ---
        results.append((pits, {
            "pit_combo": best['pit_combo'],
            "stints": best['stints'],
            "compounds": best['compounds'],
            "modes": best['modes'],
            "std_dev": std_dev,
            "total_time": total_time,
            "total_laps": sum(best['stints'])
        }))
        # --- New: Test all compound permutations for this stint structure ---
        test_compound_permutations_for_strategy(results[-1][1])


# --- New function: Test all compound permutations for a given strategy's stint lengths/pit structure ---
def test_compound_permutations_for_strategy(best_strategy):
    """
    For a given best strategy (as dict), test all permutations of compound assignments
    for the same number of stints, with fixed pit stops and modes.
    Prints the compound combination with the lowest total time.
    """
    from itertools import product
    stint_lens = list(best_strategy["stints"])
    pit_combo = best_strategy["pit_combo"]
    modes = best_strategy["modes"]
    num_stints = len(stint_lens)
    # Only consider available compounds
    compounds_available = list(tire_exprs.keys())
    # All possible compound assignments (with replacement, e.g. soft, soft, hard is allowed)
    all_compound_combos = list(product(compounds_available, repeat=num_stints))
    best_time = float('inf')
    best_combo = None
    best_modes = None
    best_stints = None
    best_pits = None
    for combo in all_compound_combos:
        # Skip combos that do not use at least two different compounds
        if len(set(combo)) < 2:
            continue
        legal = True
        for stint_len, compound, mode in zip(stint_lens, combo, modes):
            if mode not in allowed_mode_cache.get((compound, stint_len), []):
                legal = False
                break
        if not legal:
            continue
        # Compute total time
        total_time = len(pit_combo) * average_pit_time
        mode_time_multiplier = {'neutral': 1.0, 'aggressive': 0.9, 'durable': 1.1}
        for stint_len, compound, mode in zip(stint_lens, combo, modes):
            f_idx = func_index_map[tire_exprs[compound]]
            mode_func = get_mode_func(f_idx, mode)
            laps = np.arange(0, stint_len)
            stint_times = mode_func(laps) * mode_time_multiplier[mode]
            total_time += np.sum(stint_times)
        if total_time < best_time:
            best_time = total_time
            best_combo = combo
            best_modes = modes
            best_stints = stint_lens
            best_pits = pit_combo
    if best_combo is not None:
        print(f"\n  >>> Compound permutation with lowest total time for this stint structure:")
        print(f"      Pit Stops: {best_pits}")
        print(f"      Stints: {best_stints}")
        print(f"      Compounds: {best_combo}")
        print(f"      Modes: {best_modes}")
        print(f"      Total Race Time: {best_time:.2f} s")
        # Update the last element in the global results list with the new best combo
        # Retain pit_combo, stints, modes, std_dev, total_laps; update compounds and total_time
        if results:
            # Compute std_dev for the new combo (same as before, using new compounds)
            total_variance = 0.0
            for stint_len, compound in zip(best_stints, best_combo):
                compound_sigma = compound_std_map.get(compound, 0.0)
                total_variance += stint_len * (compound_sigma ** 2)
            std_dev = math.sqrt(total_variance)
            total_laps = sum(best_stints)
            results[-1] = (
                results[-1][0],
                {
                    "pit_combo": best_pits,
                    "stints": best_stints,
                    "compounds": list(best_combo),
                    "modes": list(best_modes),
                    "std_dev": std_dev,
                    "total_time": best_time,
                    "total_laps": total_laps
                }
            )
    else:
        print("\n  ⚠️ No legal compound permutation found for this stint structure.")


def main():
    """
    Main execution order for the script.
    """
    # 1. Print initial optimal strategies for each pit stop count
    print_initial_optimal_strategies(strategy_data_array)

    # Refine each initial optimal strategy
    # for pits, best in best_per_pits.items():
    #     refine_strategy(best, number_of_laps)

    # 2. Plot all optimal strategies (lap times)
    plot_all_optimal_strategies(results)
    # 3. Plot summary of optimal strategies (total time ± 3σ)
    plot_strategy_summary_with_std(results)
    # 4. Print simulation completion and save cache
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"\n=== Simulation Completed in {elapsed_time:.2f} seconds ===")
    with open(PRECOMPUTED_CACHE_FILE, 'wb') as f:
        pickle.dump(precomputed_time_cache, f)


# --- Refine strategy function ---
def refine_strategy(best, number_of_laps):
    """
    Try to refine a given strategy by adjusting pit stops and modes for better total time.
    """
    from itertools import product
    pit_variants = []
    # Generate pit variants with ±5 lap adjustments
    for pit_idx, pit_lap in enumerate(best['pit_combo']):
        for delta in range(-5, 6):
            new_pits = list(best['pit_combo'])
            new_lap = pit_lap + delta
            if min_lap_buffer <= new_lap <= number_of_laps - min_lap_buffer:
                new_pits[pit_idx] = new_lap
                # Ensure pits are sorted and legal
                new_pits_sorted = tuple(sorted(new_pits))
                # Check min lap between pits
                if all(new_pits_sorted[i+1] - new_pits_sorted[i] >= min_lap_between_pits for i in range(len(new_pits_sorted)-1)):
                    pit_variants.append(new_pits_sorted)
    pit_variants = list(set(pit_variants))  # unique

    possible_modes = ['durable', 'neutral', 'aggressive']
    mode_variants = list(product(possible_modes, repeat=len(best['stints'])))

    best_variant = None
    best_time = float('inf')

    for pit_combo in pit_variants:
        stints = [pit_combo[0]] + [pit_combo[i+1] - pit_combo[i] for i in range(len(pit_combo)-1)] + [number_of_laps - pit_combo[-1]]
        if any(s <= 0 for s in stints):
            continue
        for modes in mode_variants:
            total_time = len(pit_combo) * average_pit_time
            valid = True
            for stint_len, compound, mode in zip(stints, best['compounds'], modes):
                if mode not in allowed_mode_cache.get((compound, stint_len), []):
                    valid = False
                    break
                f_idx = func_index_map[tire_exprs[compound]]
                mode_func = get_mode_func(f_idx, mode)
                laps = np.arange(0, stint_len)
                total_time += np.sum(mode_func(laps))
            if valid and total_time < best_time:
                best_time = total_time
                best_variant = (pit_combo, stints, best['compounds'], modes, total_time)

    if best_variant:
        print(f"\n ------- Refined Optimal Strategy for {pits} Pit Stop(s):")
        print(f"  Pit Stops: {best_variant[0]}")
        print(f"  Stints: {best_variant[1]}")
        print(f"  Compounds: {best_variant[2]}")
        print(f"  Modes: {best_variant[3]}")
        print(f"  Total Time: {best_variant[4]:.2f} s")
    else:
        print("\n⚠️ No valid refined strategy found.")


# --- Evaluate a fixed tire layout (e.g., ["soft", "medium", "soft"]) ---
def evaluate_fixed_tire_layout(layout):
    """
    Given a fixed list of compounds (e.g., ["soft", "medium", "soft"]), check all legal pit stop combinations
    and mode assignments, and print the best total race time using the fixed layout.
    """
    from itertools import combinations, product
    print(f"\n=== Evaluating Fixed Tire Layout: {layout} ===")
    num_stints = len(layout)
    num_pits = num_stints - 1
    pit_positions = range(min_lap_buffer, number_of_laps - min_lap_buffer)
    best_result = None
    best_time = float('inf')
    best_modes = None
    best_pits = None
    best_stints = None
    for pit_combo in combinations(pit_positions, num_pits):
        # Enforce minimum laps between pit stops
        if any(pit_combo[i+1] - pit_combo[i] < min_lap_between_pits for i in range(len(pit_combo)-1)):
            continue
        stints = [pit_combo[0]] + [pit_combo[i+1] - pit_combo[i] for i in range(len(pit_combo)-1)] + [number_of_laps - pit_combo[-1]]
        if any(s <= 0 for s in stints):
            continue
        # For each stint, get allowed modes
        allowed_modes_per_stint = []
        legal = True
        for stint_len, compound in zip(stints, layout):
            allowed_modes = allowed_mode_cache.get((compound, stint_len), [])
            if not allowed_modes:
                legal = False
                break
            allowed_modes_per_stint.append(allowed_modes)
        if not legal:
            continue
        # Try all combinations of allowed modes for these stints
        for modes in product(*allowed_modes_per_stint):
            # Check if this mode assignment is legal for the whole layout
            # (Redundant, but keeps logic clear)
            valid = True
            for stint_len, compound, mode in zip(stints, layout, modes):
                if mode not in allowed_mode_cache.get((compound, stint_len), []):
                    valid = False
                    break
            if not valid:
                continue
            # Compute total time for this strategy
            total_time = num_pits * average_pit_time
            mode_time_multiplier = {'neutral': 1.0, 'aggressive': 0.9, 'durable': 1.1}
            for stint_idx, (stint_len, compound, mode) in enumerate(zip(stints, layout, modes)):
                f_idx = func_index_map[tire_exprs[compound]]
                mode_func = get_mode_func(f_idx, mode)
                laps = np.arange(0, stint_len)
                stint_times = mode_func(laps) * mode_time_multiplier[mode]
                total_time += np.sum(stint_times)
            if total_time < best_time:
                best_time = total_time
                best_modes = modes
                best_pits = pit_combo
                best_stints = stints
    if best_pits is not None:
        print(f"Best pit stops: {best_pits}")
        print(f"Stints: {best_stints}")
        print(f"Compounds: {layout}")
        print(f"Modes: {best_modes}")
        print(f"Total Race Time: {best_time:.2f} s")
    else:
        print("No legal strategy found for this layout.")

    # --- Diagnostic: Compare lap times for soft, medium, and hard tires over first 20 laps using neutral mode ---
    print("\n--- Diagnostic: Lap Time Comparison for Soft, Medium, Hard (First 20 Laps, Neutral Mode) ---")
    compounds = ["soft", "medium", "hard"]
    lap_range = np.arange(0, 20)
    mode = "neutral"
    mode_time_multiplier = {'neutral': 1.0, 'aggressive': 0.9, 'durable': 1.1}
    lap_times = {}
    for compound in compounds:
        if compound not in tire_exprs:
            print(f"  {compound.capitalize()}: Not available in model.")
            continue
        f_idx = func_index_map[tire_exprs[compound]]
        mode_func = get_mode_func(f_idx, mode)
        times = mode_func(lap_range) * mode_time_multiplier[mode]
        lap_times[compound] = times
        print(f"  {compound.capitalize()} Lap Times (laps 1-20):")
        print("   ", np.round(times, 2))
        print(f"    Total: {np.sum(times):.2f} s, Avg: {np.mean(times):.2f} s")
    # Quick summary: which is fastest over 20 laps?
    available = [(c, np.sum(t)) for c, t in lap_times.items()]
    if available:
        fastest = min(available, key=lambda x: x[1])
        print(f"\n  >> Fastest over first 20 laps: {fastest[0].capitalize()} ({fastest[1]:.2f} s)")


# --- Run all functions in order ---
if __name__ == "__main__":
    main()
    # --- Demonstrate fixed layout evaluation ---
    evaluate_fixed_tire_layout(["soft", "medium", "soft"])


