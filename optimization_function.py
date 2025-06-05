# Add import at the top
import time
# === F1 Race Strategy Optimizer ===
# This script simulates a race strategy using three stint functions and two pit stops.

# --- Global Configuration Variables ---
number_of_laps = 100             # Total number of laps in the race (max lap limit)
average_pit_time = 30.0          # Default average pit stop time (seconds)
min_lap_buffer = 5               # Laps to avoid at start/end before a pit may occur
min_lap_between_pits = 5         # Minimum laps between two pit stops


stint_limits_array = [
    [20, 30],   # Stint 1 [soft, hard]
    [65, 75],   # Stint 2 [soft, hard]
    [18, 28],   # Stint 3 [soft, hard]
]




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


# --- Step 2: Generalized Optimal Pit Strategy Calculation ---
optimal_function_indices = set()

def find_optimal_strategy(num_pits):
    global best_combo_info
    global optimal_function_indices

    from itertools import combinations_with_replacement, combinations

    if num_pits < 1 or num_pits > 4:
        raise ValueError("Number of pit stops must be between 1 and 4.")

    stint_count = num_pits + 1
    
    viable_indices = [0, 1, 2]

    stint_function_sets = [combo for combo in combinations_with_replacement(viable_indices, stint_count) if len(set(combo)) >= 2]

    all_stint_funcs = [
        lambdify(u, stint1_func.subs(x, u), "numpy"),
        lambdify(u, stint2_func.subs(x, u), "numpy"),
        lambdify(u, stint3_func.subs(x, u), "numpy"),
    ]

    stint_limits = {
        i: {'soft': stint_limits_array[i][0], 'hard': stint_limits_array[i][1]}
        for i in range(len(stint_limits_array))
    }

    min_total_time = float("inf")
    best_pits = None
    best_funcs = None

    for combo in stint_function_sets:
        funcs = [all_stint_funcs[i] for i in combo]
        if num_pits == 4:
            from itertools import product

            pit_range = range(min_lap_buffer, number_of_laps - min_lap_buffer, 2)
            possible_pits = product(pit_range, repeat=num_pits)
        else:
            possible_pits = combinations(range(min_lap_buffer, number_of_laps - min_lap_buffer), num_pits)

        for pits in possible_pits:
            if num_pits == 4:
                if sorted(pits) != list(pits):
                    continue
                if any((p2 - p1) < min_lap_between_pits for p1, p2 in zip(pits, pits[1:])):
                    continue
            else:
                if any((p2 - p1) < min_lap_between_pits for p1, p2 in zip(pits, pits[1:])):
                    continue

            lap_ranges = [0] + list(pits) + [number_of_laps]
            stint_laps = [np.arange(lap_ranges[i+1] - lap_ranges[i]) for i in range(stint_count)]

            total_time = 0
            valid = True
            for f_idx, (f, st) in zip(combo, zip(funcs, stint_laps)):
                stint_len = len(st)
                limits = stint_limits[f_idx]

                if stint_len > limits['hard']:
                    valid = False
                    break

                penalty = 0
                if stint_len > limits['soft']:
                    penalty = np.sum(0.5 * (np.exp(np.arange(1, stint_len - limits['soft'] + 1) / 5) - 1))

                total_time += np.sum(f(st)) + penalty

            if not valid:
                continue

            total_time += average_pit_time * num_pits

            if total_time < min_total_time:
                min_total_time = total_time
                best_pits = pits
                best_funcs = combo

    if best_funcs is None:
        print(f"No valid strategy found for {num_pits} pit stops.")
        best_combo_info = None
        return

    if num_pits in [2, 3]:
        optimal_function_indices.update(best_funcs)

    best_combo_info = (
        *[ [stint1_func, stint2_func, stint3_func][f] for f in best_funcs ],
        *best_pits,
        min_total_time
    )
    print(f"\nOptimal {num_pits}-Pit Strategy:")
    for i, p in enumerate(best_pits, 1):
        print(f"Pit {i} at Lap {p}")
    print(f"Total Time: {min_total_time:.2f} seconds")
    print("Functions used:", ', '.join(f"Function {f+1}" for f in best_funcs))


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
    opt_time = best_combo_info[5]
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
        pit_laps = combo[pit_count + 1:-1]
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
            for j in range(len(lap_times)):
                if j != pit_idx:
                    lap_times[j].append(np.nan)
            running_total += lt
            cumulative_time.append(running_total)

        ax = axs[i]
        ax2 = ax.twinx()
        for j, lts in enumerate(lap_times):
            func_idx = combo[j] if j < len(combo) else 0
            # Only show function number, not the equation, in the legend label
            ax.plot(laps, lts, label=f'Stint {j+1} (Function {combo[j]+1}, delta t: {stint_durations[j]:.1f}s)')
        for idx, p in enumerate(pit_laps):
            ax.axvline(p + 1, linestyle='--', label=f'Pit {idx + 1}')
        ax2.plot(laps, cumulative_time, color='black', alpha=0.3, label='Cumulative')
        ax2.set_ylabel("Cumulative Time (s)")
        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines + lines2, labels + labels2)
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
