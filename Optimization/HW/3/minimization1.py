import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from tabulate import tabulate

# Define the function
def f(x):
    x1, x2 = x
    return 5*x1**2 + x2**2 + 4*x1*x2 - 14*x1 - 6*x2 + 20

# Gradient of the function
def gradient_f(x):
    x1, x2 = x
    return np.array([10*x1 + 4*x2 - 14, 2*x2 + 4*x1 - 6])

# Objective function for line search
def obj_func_for_line_search(step_size, x, gradient):
    return f(x - step_size * gradient)

# Steepest descent algorithm with exact line search
def steepest_descent(f, gradient_f, x0, threshold=1e-6, max_iter=25, alpha=0.0866):
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

    # Print table
    print(tabulate(table, headers=headers, tablefmt="grid"))
    
    # Save table to a .txt file
    with open('optimization_path.txt', 'w') as file:
        file.write(tabulate(table, headers=headers, tablefmt="grid"))
    
    print("Optimization path saved to 'optimization_path.txt'")
    
    return x, f(x), np.array(trajectory)

# Initial guess and parameters
x0 = np.array([0, 10])
threshold = 10**(-6)

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
Z = f([X1, X2])

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
