from sympy import symbols, sympify, integrate, lambdify
from scipy.optimize import minimize_scalar
import matplotlib.pyplot as plt
import numpy as np

# Constants
number_of_laps = 50        # Set default number of laps
average_pit_time = 20.0    # Set default average pit time in seconds
average_lap_time = 90.0    # Set default average lap time in seconds


# Symbolic variable
x = symbols('x')

# Input functions for each stint with defaults
stint1_expr = input("Enter the function for stint 1 (in terms of x): ") or "2*x + 3"
stint2_expr = input("Enter the function for stint 2 (in terms of x): ") or "x**2 + 5*x + 10"

# Convert to symbolic expressions
stint1_func = sympify(stint1_expr)
stint2_func = sympify(stint2_expr)

print("Stint 1 function:", stint1_func)
print("Stint 2 function:", stint2_func)

# Function to calculate total time for a given pit lap
def total_race_time(pit_lap_val):
    stint1_area = integrate(stint1_func, (x, 1, pit_lap_val))
    stint2_area = integrate(stint2_func, (x, pit_lap_val, number_of_laps))
    return float(stint1_area + stint2_area + average_pit_time)

# Recursive narrowing search for optimal pit lap
def find_optimal_pit(lap_start, lap_end, threshold=1):
    if lap_end - lap_start <= threshold:
        print(f"Final choice within threshold: Lap {lap_start}")
        return lap_start, total_race_time(lap_start)

    one_third = round(lap_start + (lap_end - lap_start) / 3)
    two_third = round(lap_start + 2 * (lap_end - lap_start) / 3)

    # Prevent infinite loop by ensuring distinct test points
    if one_third == two_third or one_third == lap_start or two_third == lap_end:
        mid = (lap_start + lap_end) // 2
        print(f"Converging to midpoint: Lap {mid}")
        return mid, total_race_time(mid)

    time_one_third = total_race_time(one_third)
    time_two_third = total_race_time(two_third)

    print(f"Comparing Lap {one_third} (time: {time_one_third:.2f}) vs Lap {two_third} (time: {time_two_third:.2f})")

    if time_one_third < time_two_third:
        print(f"Choosing range: {lap_start} to {two_third}")
        return find_optimal_pit(lap_start, two_third, threshold)
    else:
        print(f"Choosing range: {one_third} to {lap_end}")
        return find_optimal_pit(one_third, lap_end, threshold)

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
