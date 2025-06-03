from sympy import symbols, sympify, integrate, lambdify
from scipy.optimize import minimize_scalar
import matplotlib.pyplot as plt
import numpy as np

# Constants
number_of_laps = 50        # Set default number of laps
average_pit_time = 20.0    # Set default average pit time in seconds



# Symbolic variable
x = symbols('x')

# Input functions for each stint with defaults
stint1_expr = input("Enter the function for stint 1 (in terms of x): ") or "2*x + 3"
stint2_expr = input("Enter the function for stint 2 (in terms of x): ") or "-0.05*x +10"

# Convert to symbolic expressions
stint1_func = sympify(stint1_expr)
stint2_func = sympify(stint2_expr)

print("Stint 1 function:", stint1_func)
print("Stint 2 function:", stint2_func)



# Input for 5 compound tire types
compound_tire_func = {}
for i in range(1, 6):
    name = f"C{i}"
    default_expr = f"{0.1 * i}*x + {5 * i}"
    expr = input(f"Enter the function for compound tire {name} (in terms of x) [default: {default_expr}]: ") or default_expr
    compound_tire_func[name] = sympify(expr)


for i in range(1, 6):
    print(f"Compound tire C{i} function:", compound_tire_func[f"C{i}"])


# Function to calculate total time for a given pit lap
def total_race_time(pit_lap_val):
    stint1_area = integrate(stint1_func, (x, 1, pit_lap_val))
    stint2_area = integrate(stint2_func, (x, pit_lap_val, number_of_laps))
    return float(stint1_area + stint2_area + average_pit_time)

# Custom narrowing search for optimal pit lap (refine around best of center ± delta)
def find_optimal_pit(start_lap, end_lap):
    center = (start_lap + end_lap) // 2
    delta = (end_lap - start_lap) // 4

    while delta > 0:
        left = max(start_lap, center - delta)
        right = min(end_lap, center + delta)

        center_time = total_race_time(center)
        left_time = total_race_time(left)
        right_time = total_race_time(right)

        print(f"Comparing Lap {left} (time: {left_time:.2f}), Lap {center} (time: {center_time:.2f}), Lap {right} (time: {right_time:.2f})")

        if center_time <= left_time and center_time <= right_time:
            print(f"Center lap {center} is optimal in this round.")
            break
        elif left_time < right_time:
            center = left
        else:
            center = right

        delta = max(1, delta // 2)
        print(f"Narrowing search to center {center} with delta {delta}")

    print(f"Final choice: Lap {center}")
    return center, total_race_time(center)

# Find optimal pit lap using narrowing method
optimal_lap, min_time = find_optimal_pit(2, number_of_laps - 1)

# Generate plot data for visualization
pit_laps = list(range(2, number_of_laps))
total_times = [total_race_time(lap) for lap in pit_laps]


# Plotting the total time vs pit lap
plt.figure(figsize=(10, 6))
plt.plot(pit_laps, total_times, marker='o', linestyle='-')
plt.axvline(optimal_lap, color='red', linestyle='--', label=f'Optimal Pit Stop: Lap {optimal_lap}')
plt.title('Total Race Time vs Pit Stop Lap')
plt.xlabel('Pit Stop Lap')
plt.ylabel('Total Race Time')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Plotting the stint functions with the optimal pit stop marked
lap_values = np.linspace(1, number_of_laps, 500)
stint1_vals = [float(stint1_func.subs(x, val)) if val <= optimal_lap else np.nan for val in lap_values]
# Adjust stint2_vals to include pit stop time at the start of the pit lap
# Add the average pit stop time to the lap time at the optimal_lap (the start of the pit lap)
stint2_vals = []
for val in lap_values:
    if val >= optimal_lap:
        lap_time = float(stint2_func.subs(x, val))
        # Add pit stop time at the start of the pit lap (i.e., at optimal_lap)
        if int(round(val)) == optimal_lap:
            lap_time += average_pit_time
        stint2_vals.append(lap_time)
    else:
        stint2_vals.append(np.nan)

plt.figure(figsize=(10, 6))
plt.plot(lap_values, stint1_vals, label='Stint 1 Function')
plt.plot(lap_values, stint2_vals, label='Stint 2 Function')
plt.axvline(optimal_lap, color='red', linestyle='--', label=f'Pit Stop at Lap {optimal_lap}')
plt.title('Stint Functions with Optimal Pit Stop')
plt.xlabel('Lap')
plt.ylabel('Time per Lap')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
