import numpy as np
from scipy.optimize import linprog

# Coefficients of the variables in the constraints
A = np.array([
    [1, 1, 1, 1, 0, 0, 0, 0, 0, 0],  # x1 + x2 + x3 + s1 = 3
    [0, 0, 1, 0, 1, 0, 0, 0, 0, 0],  # x3 + s2 = 1
    [1, 0, 1, 0, 0, 1, 0, 0, 0, 0],  # x1 + x3 + s3 = 2
    [2, 0, 5, 0, 0, 0, 1, 0, 0, 0],  # 2x1 + 5x3 + s4 = 8
    [-7, 8, 0, 0, 0, 0, 0, 1, 0, 0], # -7x1 + 8x2 + s5 = 0
    [1, 2, -1, 0, 0, 0, 0, 0, 1, 0]  # x1 + 2x2 - x3 + s6 = 1
])

# Constants in the constraints
b = np.array([3, 1, 2, 8, 0, 1])

# Coefficients for the objective function
c = np.array([0, 0, 0, 1, 1, 1, 1, 1, 1, 1])  # Minimize s1 + s2 + s3 + s4 + s5 + s6

# Bounds for each variable (all variables are non-negative)
x_bounds = [(0, None)] * 10  # No upper bound on any variable

# Solve the LP
res = linprog(c, A_eq=A, b_eq=b, bounds=x_bounds, method='highs')

# Print the results
if res.success:
    print("Optimal solution found:")
    print("Values of variables (x1, x2, x3, s1, s2, s3, s4, s5, s6):")
    print(res.x)
    print("Objective function value (sum of slacks):", res.fun)
else:
    print("No optimal solution found. Message:", res.message)
