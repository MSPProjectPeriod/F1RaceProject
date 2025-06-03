from sympy import symbols, integrate, lambdify
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application
from scipy.optimize import minimize_scalar

# Constants
number_of_laps = 50        # Set default number of laps
average_pit_time = 20.0    # Set default average pit time in seconds
average_lap_time = 90.0    # Set default average lap time in seconds


# Symbolic variable
x = symbols('x')

# Input functions for each stint
# Example: "2*x + 3", "x**2 + 5*x + 10", "100 - 0.5*x"
stint1_expr = input("Enter the function for stint 1 (in terms of x): ")
# Example: "2*x + 3", "x**2 + 5*x + 10", "100 - 0.5*x"
stint2_expr = input("Enter the function for stint 2 (in terms of x): ")

transformations = standard_transformations + (implicit_multiplication_application,)
# Convert to symbolic expressions with transformations
stint1_func = parse_expr(stint1_expr, transformations=transformations)
stint2_func = parse_expr(stint2_expr, transformations=transformations)


print("Stint 1 function:", stint1_func)
print("Stint 2 function:", stint2_func)


# Evaluate total time for each valid pit stop lap
pit_lap = 1
while pit_lap < number_of_laps - 1:
    stint1_area = integrate(stint1_func, (x, 1, pit_lap))
    stint2_area = integrate(stint2_func, (x, pit_lap, number_of_laps))
    total_time = stint1_area + stint2_area + average_pit_time
    print(f"Pit lap: {pit_lap}, Total time: {total_time}")
    pit_lap += 1
