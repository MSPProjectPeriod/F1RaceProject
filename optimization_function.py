# === F1 Race Strategy Optimizer ===
# This script simulates a race strategy using three stint functions and two pit stops.

# --- Global Configuration Variables ---
number_of_laps = 100             # Total number of laps in the race (max lap limit)
average_pit_time = 30.0          # Default average pit stop time (seconds)
min_lap_buffer = 5               # Laps to avoid at start/end before a pit may occur
min_lap_between_pits = 5         # Minimum laps between two pit stops

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


# --- Step 2: Optimal Two-Pit Strategy Calculation ---
def find_optimal_two_pit_strategy():
    global best_combo_info

    from itertools import combinations_with_replacement

    stint_function_sets = []
    for combo in combinations_with_replacement([0, 1, 2], 3):
        if len(set(combo)) >= 2:
            stint_function_sets.append(combo)

    min_total_time = float("inf")
    best_p1, best_p2 = None, None
    best_funcs = None

    all_stint_funcs = [
        lambdify(u, stint1_func.subs(x, u), "numpy"),
        lambdify(u, stint2_func.subs(x, u), "numpy"),
        lambdify(u, stint3_func.subs(x, u), "numpy"),
    ]

    for combo in stint_function_sets:
        f1_idx, f2_idx, f3_idx = combo
        funcs = [all_stint_funcs[f1_idx], all_stint_funcs[f2_idx], all_stint_funcs[f3_idx]]

        for p1 in range(min_lap_buffer, number_of_laps - min_lap_buffer):
            for p2 in range(p1 + min_lap_between_pits, number_of_laps - min_lap_buffer):
                stint1_laps = np.arange(0, p1)
                stint2_laps = np.arange(0, p2 - p1)
                stint3_laps = np.arange(0, number_of_laps - p2)

                total_time = (
                    np.sum(funcs[0](stint1_laps)) +
                    average_pit_time +
                    np.sum(funcs[1](stint2_laps)) +
                    average_pit_time +
                    np.sum(funcs[2](stint3_laps))
                )

                if total_time < min_total_time:
                    min_total_time = total_time
                    best_p1, best_p2 = p1, p2
                    best_funcs = combo

    best_combo_info = (
        [stint1_func, stint2_func, stint3_func][best_funcs[0]],
        [stint1_func, stint2_func, stint3_func][best_funcs[1]],
        [stint1_func, stint2_func, stint3_func][best_funcs[2]],
        best_p1,
        best_p2,
        min_total_time
    )
    print("\nOptimal Two-Pit Strategy:")
    print(f"Pit 1 at Lap {best_p1}, Pit 2 at Lap {best_p2}, Total Time: {min_total_time:.2f} seconds")
    print("Functions used: Function {}, Function {}, Function {}".format(*[f+1 for f in best_funcs]))


find_optimal_two_pit_strategy()

# --- 3D Visualization of Pit Strategy Surface ---
from mpl_toolkits.mplot3d import Axes3D

def visualize_pit_strategy_surface():
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    pit1_range = range(min_lap_buffer, number_of_laps - min_lap_buffer)
    pit2_range = range(min_lap_buffer + min_lap_between_pits, number_of_laps - min_lap_buffer)

    X, Z, Y = [], [], []

    # Prepare lambdified functions again
    stint_funcs = [
        lambdify(u, stint1_func.subs(x, u), "numpy"),
        lambdify(u, stint2_func.subs(x, u), "numpy"),
        lambdify(u, stint3_func.subs(x, u), "numpy"),
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

    # Draw red lines to axes
    ax.plot([opt_p1, opt_p1], [opt_p2, opt_p2], [0, opt_time], color='red', linestyle='--')
    ax.plot([opt_p1, opt_p1], [0, opt_p2], [opt_time, opt_time], color='red', linestyle='--')
    ax.plot([0, opt_p1], [opt_p2, opt_p2], [opt_time, opt_time], color='red', linestyle='--')

    # Annotate optimal values (offset labels to reduce overlap)
    ax.text(opt_p1, opt_p2 - 3, 0, f"Pit 1: Lap {opt_p1}", color='red')
    ax.text(opt_p1 - 3, opt_p2, 0, f"Pit 2: Lap {opt_p2}", color='red')
    ax.text(opt_p1, opt_p2, opt_time, f"\nTime: {opt_time:.2f}", color='red')

    ax.set_xlabel("Pit 1 Lap")
    ax.set_ylabel("Pit 2 Lap")
    ax.set_zlabel("Total Race Time (s)")
    ax.set_title("Race Time by Pit Stop Strategy")
    ax.legend()
    plt.tight_layout()
    plt.show()


visualize_pit_strategy_surface()

# --- Plot Lap Times with Optimal Pit Strategy ---
def plot_stint_functions_with_pits():
    if best_combo_info is None:
        print("No optimal strategy available to plot.")
        return

    stint1_func, stint2_func, stint3_func, pit1, pit2, _ = best_combo_info

    # Define lap range
    laps = np.arange(1, number_of_laps + 1)
    lap_times_stint1 = []
    lap_times_stint2 = []
    lap_times_stint3 = []
    cumulative_time = []

    running_total = 0.0

    for lap in laps:
        if lap <= pit1:
            x_val = lap
            lap_time = float(stint1_func.subs(x, x_val))
            lap_times_stint1.append(lap_time)
        else:
            lap_times_stint1.append(np.nan)

        if pit1 < lap <= pit2:
            x_val = lap - pit1
            lap_time = float(stint2_func.subs(x, x_val))
            if lap == pit1 + 1:
                lap_time += average_pit_time
            lap_times_stint2.append(lap_time)
        else:
            lap_times_stint2.append(np.nan)

        if lap > pit2:
            x_val = lap - pit2
            lap_time = float(stint3_func.subs(x, x_val))
            if lap == pit2 + 1:
                lap_time += average_pit_time
            lap_times_stint3.append(lap_time)
        else:
            lap_times_stint3.append(np.nan)

        # Calculate cumulative time for this lap
        current_lap_time = (
            (lap_times_stint1[-1] if not np.isnan(lap_times_stint1[-1]) else 0) +
            (lap_times_stint2[-1] if not np.isnan(lap_times_stint2[-1]) else 0) +
            (lap_times_stint3[-1] if not np.isnan(lap_times_stint3[-1]) else 0)
        )
        running_total += current_lap_time
        cumulative_time.append(running_total)

    # Calculate total time per stint
    total_time_s1 = np.nansum(lap_times_stint1)
    total_time_s2 = np.nansum(lap_times_stint2)
    total_time_s3 = np.nansum(lap_times_stint3)

    # Determine legend labels based on function equality
    f1 = stint1_func
    f2 = stint2_func
    f3 = stint3_func

    if f1 == f2 == f3:
        labels = ['Function A']
        # Plot all three with same label, but only label once
        plot_labels = [labels[0], None, None]
    elif f1 == f2 and f2 != f3:
        labels = ['Function A', 'Function A (reused)', 'Function B']
        plot_labels = labels
    elif f2 == f3 and f1 != f2:
        labels = ['Function A', 'Function B', 'Function B (reused)']
        plot_labels = labels
    else:
        labels = ['Function 1', 'Function 2', 'Function 3']
        plot_labels = labels

    # Plotting
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Adjusted colors for stints
    ax1.plot(laps, lap_times_stint1, label=plot_labels[0], color='blue')
    ax1.plot(laps, lap_times_stint2, label=plot_labels[1], color='darkorange')
    ax1.plot(laps, lap_times_stint3, label=plot_labels[2], color='crimson')
    # Adjusted pit stop line colors with lap numbers in legend
    ax1.axvline(pit1 + 1, color='darkred', linestyle='--', label=f'1st Pit Stop (Lap {pit1 + 1})')
    ax1.axvline(pit2 + 1, color='goldenrod', linestyle='--', label=f'2nd Pit Stop (Lap {pit2 + 1})')
    ax1.set_xlabel("Lap")
    ax1.set_ylabel("Lap Time (s)")
    ax1.set_title("Lap Times with Optimal Pit Strategy")
    ax1.legend(loc='upper left')
    ax1.grid(True)

    ax1.text(pit1 + (pit2 - pit1) / 2, max(lap_times_stint2) + 2, f"{total_time_s2:.1f}s", color='darkorange', ha='center')
    ax1.text(pit2 + (number_of_laps - pit2) / 2, max(lap_times_stint3) + 2, f"{total_time_s3:.1f}s", color='crimson', ha='center')

    ax2 = ax1.twinx()
    # Adjusted cumulative time line color to black
    ax2.plot(laps, cumulative_time, color='black', alpha=0.4, label='Cumulative Race Time')
    ax2.fill_between(laps, cumulative_time, color='black', alpha=0.05)
    ax2.set_ylabel("Cumulative Time (s)")
    ax2.legend(loc='upper right')

    # Display delta times in a text box
    delta_text = (
        f"Δt1 (Stint 1): {total_time_s1:.1f}s\n"
        f"Δt2 (Stint 2): {total_time_s2:.1f}s\n"
        f"Δt3 (Stint 3): {total_time_s3:.1f}s"
    )
    props = dict(boxstyle='round', facecolor='white', alpha=0.8)
    ax2.text(0.97, 0.25, delta_text, transform=ax2.transAxes, fontsize=10,
             verticalalignment='top', horizontalalignment='right', bbox=props)

    plt.tight_layout()
    plt.show()

plot_stint_functions_with_pits()
