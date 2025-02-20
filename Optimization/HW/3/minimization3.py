import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from tabulate import tabulate

# Define the function
def f(x):
    Q = np.array([[20, 5], [5, 16]])
    C = np.array([14, 6])
    return 0.5 * x.T @ Q @ x - C.T @ x + 10

# Gradient of the function
def gradient_f(x):
    Q = np.array([[20, 5], [5, 2]])
    C = np.array([14, 6])
    return Q @ x - C

# Objective function for line search
def obj_func_for_line_search(step_size, x, gradient):
    return f(x - step_size * gradient)

# Steepest descent algorithm with exact line search
def steepest_descent(f, gradient_f, x0, threshold=1e-6, max_iter=10000, alpha=0.0866):
    headers = ["Iteration", "x1", "x2", "Gradient", "Function Value", "Step Size"]
    table = []
    
    x = x0
    trajectory = [x]
    
    for k in range(max_iter):
        grad = gradient_f(x)
        function_value = f(x)
        
        row = [k, x[0], x[1], grad, function_value, '-']
        table.append(row)

        # Check termination condition
        if np.linalg.norm(grad) < threshold:
            break
        
        # Exact line search for step size
        step_size = minimize(obj_func_for_line_search, alpha, args=(x, grad)).x[0]
        x = x - step_size * grad
        row[-1] = step_size
        
        trajectory.append(x)

    print(tabulate(table[:10], headers=headers, tablefmt="grid"))  # First 10 iterations
    print("...\n")
    print(tabulate(table[-10:], headers=headers, tablefmt="grid"))  # Last 10 iterations
    return x, f(x), np.array(trajectory)

# Initial guess and parameters
x0 = np.array([40, -100])
threshold = 1e-6  # Stop criterion ε = 10^-6

# Perform optimization with steepest descent
optimal_point_sd, min_value_sd, trajectory = steepest_descent(f, gradient_f, x0, threshold=threshold)

print("\nOptimal point (Steepest Descent):", optimal_point_sd)
print("Minimum value (Steepest Descent):", min_value_sd)

# Calculate the range of trajectory points with a buffer
buffer = 2  # Adjust the buffer size as needed
trajectory_range = np.abs(trajectory - optimal_point_sd).max() + buffer

# Plotting the contours of the function with increased bounds
x1 = np.linspace(optimal_point_sd[0] - trajectory_range, optimal_point_sd[0] + trajectory_range, 100)
x2 = np.linspace(optimal_point_sd[1] - trajectory_range, optimal_point_sd[1] + trajectory_range, 100)
X1, X2 = np.meshgrid(x1, x2)

# Calculate the function values for each point on the grid
Z = np.zeros_like(X1)
for i in range(len(x1)):
    for j in range(len(x2)):
        Z[i, j] = f(np.array([X1[i, j], X2[i, j]]))

# Calculate the function values along the optimization path
function_values = [f(point) for point in trajectory]

# Plotting the contours of the function with increasing levels
levels = np.linspace(min(function_values), max(function_values), 20)

plt.figure(figsize=(8, 6))
plt.contour(X1, X2, Z, levels=levels)

# Plotting the optimization path
plt.plot(trajectory[:, 0], trajectory[:, 1], marker='o', color='red', label='Optimization Path')

# Plotting the optimal point
plt.scatter(optimal_point_sd[0], optimal_point_sd[1], color='black', marker='x', label='Optimal Point')

plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Optimization Path of Steepest Descent')
plt.legend()
plt.grid(True)
plt.show()
