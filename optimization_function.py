import time
# === F1 Race Strategy Optimizer ===
# This script simulates a race strategy using three stint functions and two pit stops.

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


# --- Input and Parse Stint Functions ---
# Each function is defined in terms of x (laps since the start of the stint)
stint1_expr = input("Enter the function for stint 1 (in terms of x) [0.02*x**2 - 0.5*x + 88]: ") or "0.02*x**2 - 0.5*x + 88"
stint2_expr = input("Enter the function for stint 2 (in terms of x) [-0.1*x + 85]: ") or "-0.1*x + 85"
stint3_expr = input("Enter the function for stint 3 (in terms of x) [0.03*x**2 + 0.2*x + 82]: ") or "0.03*x**2 + 0.2*x + 82"

# Convert to symbolic expressions
stint1_func = sympify(stint1_expr)
stint2_func = sympify(stint2_expr)
stint3_func = sympify(stint3_expr)

print("Stint 1 function:", stint1_func)
print("Stint 2 function:", stint2_func)
print("Stint 3 function:", stint3_func)

# --- Stint mode settings (moved here after stint1_func, stint2_func, stint3_func are defined) ---
stint_base_exprs = [stint1_func, stint2_func, stint3_func]
stint_mode_settings = {}

for i, base_expr in enumerate(stint_base_exprs):
    f_prime = diff(base_expr, x)
    stint_mode_settings[i] = {
        'neutral': base_expr,
        'aggressive': base_expr - mode_alpha * base_expr,
        'durable': base_expr + mode_alpha * base_expr
    }


# --- Step 2: Generalized Optimal Pit Strategy Calculation ---
optimal_function_indices = set()

def find_optimal_strategy(num_pits):
    global best_combo_info
    global optimal_function_indices

    from itertools import combinations_with_replacement, combinations, product

    if num_pits < 1 or num_pits > 4:
        raise ValueError("Number of pit stops must be between 1 and 4.")

    stint_count = num_pits + 1
    
    viable_indices = [0, 1, 2]

    stint_function_sets = [combo for combo in combinations_with_replacement(viable_indices, stint_count) if len(set(combo)) >= 2]

    mode_options = ['neutral', 'aggressive', 'durable']
    mode_combinations = list(product(mode_options, repeat=stint_count))

    all_stint_funcs = [
        lambdify(u, stint1_func.subs(x, u), "numpy"),
        lambdify(u, stint2_func.subs(x, u), "numpy"),
        lambdify(u, stint3_func.subs(x, u), "numpy"),
    ]

    min_total_time = float("inf")
    func_mode_cache = {}
    # Precompute penalty cache for up to max_penalty_len
    max_penalty_len = number_of_laps
    penalty_cache = np.cumsum(np.exp(np.arange(1, max_penalty_len + 1) / 5) - 1)
    best_pits = None
    best_funcs = None
    best_modes = None
    lap_time_cache = {}

    # Pre-filter valid pit combinations outside the main loop
    if num_pits == 4:
        pit_range = range(min_lap_buffer, number_of_laps - min_lap_buffer, 2)
        all_possible_pits = list(product(pit_range, repeat=num_pits))
        valid_pit_combinations = [
            pits for pits in all_possible_pits
            if sorted(pits) == list(pits) and all((p2 - p1) >= min_lap_between_pits for p1, p2 in zip(pits, pits[1:]))
        ]
    else:
        all_possible_pits = list(combinations(range(min_lap_buffer, number_of_laps - min_lap_buffer), num_pits))
        valid_pit_combinations = [
            pits for pits in all_possible_pits
            if all((p2 - p1) >= min_lap_between_pits for p1, p2 in zip(pits, pits[1:]))
        ]

    # === Stage 1: Find best function order and pit stops using only neutral mode ===
    stage1_min_time = float("inf")
    stage1_best_funcs = None
    stage1_best_pits = None
    stage1_best_modes = None  # always 'neutral'
    for combo in stint_function_sets:
        stint_limits = {}
        adjusted_funcs = {}
        for i in range(len(combo)):
            f_idx = combo[i]
            mode = 'neutral'
            stint_limits[f_idx] = {
                'soft': stint_limits_array[f_idx][0],
                'hard': stint_limits_array[f_idx][1]
            }
            if (f_idx, mode) not in func_mode_cache:
                # Use mode-adjusted expressions
                adjusted_expr = stint_mode_settings[f_idx][mode]
                func_mode_cache[(f_idx, mode)] = lambdify(u, adjusted_expr.subs(x, u), "numpy")
            adjusted_funcs[f_idx] = func_mode_cache[(f_idx, mode)]
        funcs = [adjusted_funcs[combo[i]] for i in range(len(combo))]
        for pits in valid_pit_combinations:
            lap_ranges = [0] + list(pits) + [number_of_laps]
            stint_laps = [np.arange(lap_ranges[i+1] - lap_ranges[i]) for i in range(stint_count)]
            total_time = 0
            valid = True
            for stint_idx, (f_idx, f, st) in enumerate(zip(combo, funcs, stint_laps)):
                stint_len = len(st)
                # --- Adjusted stint length for driving mode ---
                mode_factor = {'neutral': 1.0, 'aggressive': 1.1, 'durable': 0.9}
                mode = 'neutral'
                adjusted_stint_len = stint_len * mode_factor[mode]
                limits = stint_limits[f_idx]
                if adjusted_stint_len > limits['hard']:
                    valid = False
                    break
                penalty = 0
                if adjusted_stint_len > limits['soft']:
                    penalty_factor = penalty_growth_factors[f_idx]
                    over_soft = int(np.ceil(adjusted_stint_len - limits['soft']))
                    penalty = penalty_factor * penalty_cache[over_soft - 1]
                key = (f_idx, 'neutral', stint_len)
                if key in lap_time_cache:
                    stint_time = lap_time_cache[key]
                else:
                    stint_time = np.sum(f(st)) + penalty
                    lap_time_cache[key] = stint_time
                total_time += stint_time
            if not valid:
                continue
            total_time += average_pit_time * num_pits
            if total_time < stage1_min_time:
                stage1_min_time = total_time
                stage1_best_funcs = combo
                stage1_best_pits = pits
                stage1_best_modes = tuple(['neutral'] * stint_count)

    if stage1_best_funcs is None:
        print(f"No valid strategy found for {num_pits} pit stops.")
        best_combo_info = None
        return

    # === Stage 2: Optimize stint modes for the best function/pit combo ===
    min_total_time = float("inf")
    best_funcs = stage1_best_funcs
    best_pits = stage1_best_pits
    best_modes = None
    lap_time_cache_modes = {}
    for mode_combo in mode_combinations:
        stint_limits = {}
        adjusted_funcs = {}
        for i in range(len(best_funcs)):
            f_idx = best_funcs[i]
            mode = mode_combo[i]
            stint_limits[f_idx] = {
                'soft': stint_limits_array[f_idx][0],
                'hard': stint_limits_array[f_idx][1]
            }
            if (f_idx, mode) not in func_mode_cache:
                adjusted_expr = stint_mode_settings[f_idx][mode]
                func_mode_cache[(f_idx, mode)] = lambdify(u, adjusted_expr.subs(x, u), "numpy")
            adjusted_funcs[f_idx] = func_mode_cache[(f_idx, mode)]
        funcs = [adjusted_funcs[best_funcs[i]] for i in range(len(best_funcs))]
        lap_ranges = [0] + list(best_pits) + [number_of_laps]
        stint_laps = [np.arange(lap_ranges[i+1] - lap_ranges[i]) for i in range(stint_count)]
        total_time = 0
        valid = True
        for stint_idx, (f_idx, f, st) in enumerate(zip(best_funcs, funcs, stint_laps)):
            stint_len = len(st)
            # --- Adjusted stint length for driving mode ---
            mode_factor = {'neutral': 1.0, 'aggressive': 1.1, 'durable': 0.9}
            mode = mode_combo[stint_idx] if 'mode_combo' in locals() else 'neutral'
            adjusted_stint_len = stint_len * mode_factor[mode]
            limits = stint_limits[f_idx]
            if adjusted_stint_len > limits['hard']:
                valid = False
                break
            penalty = 0
            if adjusted_stint_len > limits['soft']:
                penalty_factor = penalty_growth_factors[f_idx]
                over_soft = int(np.ceil(adjusted_stint_len - limits['soft']))
                penalty = penalty_factor * penalty_cache[over_soft - 1]
            key = (f_idx, mode_combo[stint_idx], stint_len)
            if key in lap_time_cache_modes:
                stint_time = lap_time_cache_modes[key]
            else:
                stint_time = np.sum(f(st)) + penalty
                lap_time_cache_modes[key] = stint_time
            total_time += stint_time
        if not valid:
            continue
        total_time += average_pit_time * num_pits
        if total_time < min_total_time:
            min_total_time = total_time
            best_modes = mode_combo

    if num_pits in [2, 3]:
        optimal_function_indices.update(best_funcs)

    # === Stage 3: Mid-Stint Mode Switch Optimization ===
    # Allow for multiple sequential mode changes per stint, testing all possible mode combinations (any order, repeated or non-consecutive)
    improved_total_time = min_total_time
    improved_modes = list(best_modes)

    mode_order = ['aggressive', 'neutral', 'durable']
    mode_factor = {'neutral': 1.0, 'aggressive': 1.1, 'durable': 0.9}
    from itertools import product

    for stint_idx in range(len(best_funcs)):
        f_idx = best_funcs[stint_idx]
        initial_mode = best_modes[stint_idx]
        current_mode_idx = mode_order.index(initial_mode)
        lap_ranges = [0] + list(best_pits) + [number_of_laps]
        stint_start = lap_ranges[stint_idx]
        stint_end = lap_ranges[stint_idx + 1]
        stint_len = stint_end - stint_start
        soft_limit = stint_limits_array[f_idx][0]
        # Only consider switching if stint_len < soft_limit and not already most aggressive
        if stint_len < soft_limit and current_mode_idx < len(mode_order) - 1:
            best_local_time = None
            best_local_mode = None
            # For each possible number of mode segments (from 2 up to max possible)
            for num_segments in range(2, len(mode_order) - current_mode_idx + 1):
                # Allow any combination and order of modes, including repeated and non-consecutive transitions
                possible_mode_sequences = list(product(mode_order, repeat=num_segments))
                valid_mode_seqs = list(possible_mode_sequences)
                # Always test full-aggressive as fallback (single mode segment)
                valid_mode_seqs.append((mode_order[-1],))
                # Generate all valid ways to split stint_len into num_segments, each segment at least 1 lap
                def gen_splits(total, parts):
                    if parts == 1:
                        if total >= 1:
                            yield (total,)
                        return
                    for i in range(1, total - parts + 2):
                        for rest in gen_splits(total - i, parts - 1):
                            yield (i,) + rest
                for mode_seq in valid_mode_seqs:
                    for split in gen_splits(stint_len, len(mode_seq)):
                        # Calculate time for this mode sequence and split
                        total_time_stint = 0
                        adjusted_stint_len = 0
                        segment_desc = []
                        lap_cursor = 0
                        for mode, seg_len in zip(mode_seq, split):
                            f_mode = func_mode_cache[(f_idx, mode)]
                            laps = np.arange(seg_len)
                            time_seg = np.sum(f_mode(laps))
                            total_time_stint += time_seg
                            adjusted_stint_len += seg_len * mode_factor[mode]
                            segment_desc.append(mode)
                            lap_cursor += seg_len
                        # Penalty check
                        penalty = 0
                        if adjusted_stint_len > stint_limits_array[f_idx][0]:
                            over_soft = int(np.ceil(adjusted_stint_len - stint_limits_array[f_idx][0]))
                            penalty = penalty_growth_factors[f_idx] * penalty_cache[over_soft - 1]
                        total_time_stint += penalty
                        # Recalculate race time
                        total_time_race = 0
                        for j, (f_j, m_j) in enumerate(zip(best_funcs, improved_modes)):
                            if j == stint_idx:
                                total_time_race += total_time_stint
                            else:
                                key = (f_j, m_j, lap_ranges[j+1] - lap_ranges[j])
                                if key in lap_time_cache_modes:
                                    total_time_race += lap_time_cache_modes[key]
                                else:
                                    # Recompute the total time for this mode segment if missing from cache
                                    if '→' in m_j:
                                        segment_modes2 = m_j.split('@-')[0].split('→')
                                        segment_lengths2 = list(map(int, m_j.split('@-(')[1][:-1].split(',')))
                                        laps_cursor2 = 0
                                        for mode2, seg_len2 in zip(segment_modes2, segment_lengths2):
                                            f_mode2 = func_mode_cache[(f_j, mode2)]
                                            laps2 = np.arange(seg_len2)
                                            total_time_race += np.sum(f_mode2(laps2))
                                    else:
                                        f_mode2 = func_mode_cache[(f_j, m_j)]
                                        laps2 = np.arange(lap_ranges[j+1] - lap_ranges[j])
                                        total_time_race += np.sum(f_mode2(laps2))
                        total_time_race += average_pit_time * num_pits
                        # Mode description: e.g., 'durable→neutral→aggressive@-(10,5,3)'
                        if len(set(mode_seq)) == 1:
                            mode_desc = mode_seq[0]
                        else:
                            mode_desc = '→'.join(mode_seq) + '@-(' + ','.join(str(x) for x in split) + ')'
                        if best_local_time is None or total_time_race < best_local_time:
                            best_local_time = total_time_race
                            best_local_mode = mode_desc
            if best_local_time is not None and best_local_time < improved_total_time:
                improved_total_time = best_local_time
                improved_modes[stint_idx] = best_local_mode

    best_combo_info = (
        *[ [stint1_func, stint2_func, stint3_func][f] for f in best_funcs ],
        *best_pits,
        tuple(improved_modes),
        improved_total_time
    )
    print(f"\nOptimal {num_pits}-Pit Strategy:")
    for i, p in enumerate(best_pits, 1):
        print(f"Pit {i} at Lap {p}")
    print(f"Total Time: {improved_total_time:.2f} seconds")
    print("Functions used:", ', '.join(f"Function {f+1}" for f in best_funcs))
    print("Modes used:", ', '.join(str(m) for m in improved_modes))


results = []

# Time and collect results for each pit count
for pit_count in range(1, 5):
    start = time.time()
    find_optimal_strategy(pit_count)
    elapsed = time.time() - start
    print(f"{pit_count}-stop strategy took {elapsed:.2f} seconds")
    results.append((pit_count, best_combo_info))

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


# --- Plot multiple strategies in 2x2 grid ---
def plot_multiple_strategies(results):
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    axs = axs.flatten()

    for i, (pit_count, combo) in enumerate(results):
        if combo is None:
            continue

        stint_funcs = combo[:pit_count + 1]
        best_funcs = [ [stint1_func, stint2_func, stint3_func].index(func) for func in stint_funcs ]
        pit_laps = combo[pit_count + 1:2*pit_count + 1]
        best_modes = combo[2*pit_count + 1]
        total_time = combo[-1]

        stint_durations = [0.0] * len(stint_funcs)

        laps = np.arange(1, number_of_laps + 1)
        cumulative_time = []
        lap_times = [[] for _ in stint_funcs]
        running_total = 0.0

        for lap in laps:
            pit_idx = next((j for j, p in enumerate(pit_laps) if lap <= p), len(pit_laps))
            if pit_idx > 0:
                stint_start = pit_laps[pit_idx - 1]
            else:
                stint_start = 0
            x_val = lap - stint_start
            lt = float(stint_funcs[pit_idx].subs(x, x_val))
            if lap in [p + 1 for p in pit_laps]:
                lt += average_pit_time
            stint_durations[pit_idx] += lt
            lap_times[pit_idx].append(lt)
            for j, lts in enumerate(lap_times):
                if j != pit_idx:
                    lap_times[j].append(np.nan)
            running_total += lt
            cumulative_time.append(running_total)

        ax = axs[i]
        ax2 = ax.twinx()
        for j, lts in enumerate(lap_times):
            ax.plot(laps, lts, label=f'Stint {j+1} (Function {best_funcs[j]+1}, Mode {best_modes[j]}, Δt: {stint_durations[j]:.1f}s)')
        for idx, p in enumerate(pit_laps):
            ax.axvline(p + 1, linestyle='--', label=f'Pit {idx + 1}')
        # Fill cumulative time by stint mode using updated colors
        colors = {
            'neutral': '#40e0d0',     # Turquoise
            'aggressive': '#ffd700',  # Yellow
            'durable': '#1f77b4'      # Blue
        }
        stint_ranges = [0] + [p + 1 for p in pit_laps] + [number_of_laps + 1]
        # For each stint, determine mode transitions and fill between them
        for j in range(len(stint_ranges) - 1):
            stint_start = stint_ranges[j]
            stint_end = stint_ranges[j+1]
            mode_desc = best_modes[j]
            # If the mode includes transitions, fill each segment separately
            if '→' in mode_desc and '@-' in mode_desc:
                mode_sequence = mode_desc.split('@-')[0].split('→')
                segment_lengths = list(map(int, mode_desc.split('@-(')[1][:-1].split(',')))
                lap_cursor = stint_start
                for mode, seg_len in zip(mode_sequence, segment_lengths):
                    seg_start = lap_cursor
                    seg_end = lap_cursor + seg_len
                    # Ensure the segment is within the valid lap range
                    seg_start_idx = max(seg_start, 0)
                    seg_end_idx = min(seg_end, number_of_laps)
                    if seg_end_idx > seg_start_idx:
                        ax2.fill_between(
                            laps[seg_start_idx:seg_end_idx+1],
                            cumulative_time[seg_start_idx:seg_end_idx+1],
                            y2=ax2.get_ylim()[0],
                            color=colors.get(mode, 'black'),
                            alpha=0.1,
                            step="pre"
                        )
                    lap_cursor = seg_end
            else:
                # Single mode for this stint
                seg_start = stint_start
                seg_end = stint_end
                seg_start_idx = max(seg_start, 0)
                seg_end_idx = min(seg_end, number_of_laps)
                if seg_end_idx > seg_start_idx:
                    ax2.fill_between(
                        laps[seg_start_idx:seg_end_idx+1],
                        cumulative_time[seg_start_idx:seg_end_idx+1],
                        y2=ax2.get_ylim()[0],
                        color=colors.get(mode_desc, 'black'),
                        alpha=0.1,
                        step="pre"
                    )
        # Add vertical lines for mode transitions within stints
        for j in range(len(stint_ranges) - 1):
            start = stint_ranges[j]
            mode_desc = best_modes[j]
            if '→' in mode_desc and '@-' in mode_desc:
                # Parse mode sequence and segment lengths
                mode_sequence = mode_desc.split('@-')[0].split('→')
                segment_lengths = list(map(int, mode_desc.split('@-(')[1][:-1].split(',')))
                lap_cursor = start
                for mode, seg_len in zip(mode_sequence[:-1], segment_lengths[:-1]):
                    lap_cursor += seg_len
                    # The next mode is mode_sequence[mode_sequence.index(mode)+1], but zip ensures order
                    next_mode = mode_sequence[mode_sequence.index(mode)+1]
                    ax.axvline(lap_cursor, linestyle='-', color=colors.get(next_mode, 'black'), alpha=0.7)
        ax2.plot(laps, cumulative_time, color='black', alpha=0.3, label='Cumulative')
        ax2.set_ylabel("Cumulative Time (s)")
        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines + lines2, labels + labels2, loc='upper left')
        ax.set_title(f"{pit_count} Pit Stop(s) - Total Time: {total_time:.1f}s")
        ax.set_xlabel("Lap")
        ax.set_ylabel("Lap Time (s)")
        ax.grid(True)

    plt.tight_layout()
    plt.show()

plot_multiple_strategies(results)
# Only visualize the 2-pit strategy surface if it exists
if results[1][1] is not None:
    best_combo_info = results[1][1]
    visualize_pit_strategy_surface()

# --- Option 1: 3-pit 4D visualization ---
def visualize_3_pit_strategy_4d():
    if results[2][1] is None:
        print("No valid 3-pit strategy to visualize.")
        return

    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    p1_range = range(min_lap_buffer, number_of_laps - 3 * min_lap_between_pits)
    p2_range = range(min_lap_buffer + min_lap_between_pits, number_of_laps - 2 * min_lap_between_pits)
    p3_range = range(min_lap_buffer + 2 * min_lap_between_pits, number_of_laps - min_lap_buffer)

    stint_funcs = [
        lambdify(u, results[2][1][0].subs(x, u), "numpy"),
        lambdify(u, results[2][1][1].subs(x, u), "numpy"),
        lambdify(u, results[2][1][2].subs(x, u), "numpy"),
        lambdify(u, results[2][1][3].subs(x, u), "numpy"),
    ]

    X, Y, Z, C = [], [], [], []

    for p1 in p1_range:
        for p2 in p2_range:
            for p3 in p3_range:
                if not (p1 < p2 < p3):
                    continue
                if (p2 - p1) < min_lap_between_pits or (p3 - p2) < min_lap_between_pits:
                    continue

                lap_bounds = [0, p1, p2, p3, number_of_laps]
                lap_segments = [np.arange(lap_bounds[i+1] - lap_bounds[i]) for i in range(4)]

                total_time = 0
                for f, laps in zip(stint_funcs, lap_segments):
                    total_time += np.sum(f(laps))
                total_time += 3 * average_pit_time

                X.append(p1)
                Y.append(p2)
                Z.append(p3)
                C.append(total_time)

    sc = ax.scatter(X, Y, Z, c=C, cmap='viridis', s=10)
    fig.colorbar(sc, ax=ax, label="Total Race Time (s)")
    ax.set_xlabel("Pit 1 Lap")
    ax.set_ylabel("Pit 2 Lap")
    ax.set_zlabel("Pit 3 Lap")
    ax.set_title("3-Pit Strategy Visualization (Color = Total Time)")

    # Highlight optimal pit stop combination
    opt_p1, opt_p2, opt_p3 = results[2][1][4:7]
    opt_time = results[2][1][-1]
    ax.scatter(opt_p1, opt_p2, opt_p3, c='red', s=100, label='Optimal Pit Stops')
    ax.text(opt_p1, opt_p2, opt_p3, f"({opt_p1}, {opt_p2}, {opt_p3}, {opt_time:.1f}s)",
            color='red', fontsize=9, fontweight='bold', ha='left', va='bottom')
    ax.legend()

    plt.tight_layout()
    plt.show()

visualize_3_pit_strategy_4d()
