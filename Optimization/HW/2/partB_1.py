import numpy as np
import matplotlib.pyplot as plt

# Define objective function
def objective_function(x1, x2):
    return (x1 - 3)**2 + (x2 - 3)**2

# Define constraints
def constraint_1(x1, x2):
    return 4*x1**2 + 9*x2**2 <= 36

def constraint_2(x1, x2):
    return x1**2 + 3*x2**2 <= 3

def constraint_3(x1, x2):
    return x1 <= -1

# Generate grid points
x1 = np.linspace(-5, 5, 400)
x2 = np.linspace(-5, 5, 400)
X1, X2 = np.meshgrid(x1, x2)

# Compute objective function values
Z = objective_function(X1, X2)

# Create contour plot for the objective function
plt.contourf(X1, X2, Z, levels=50, cmap='viridis')
plt.colorbar(label='Objective Function')

# Plot constraint regions
plt.contour(X1, X2, constraint_1(X1, X2), [0, 1], colors='r', linestyles='dashed')
plt.contour(X1, X2, constraint_2(X1, X2), [0, 1], colors='g', linestyles='dashed')
plt.contour(X1, X2, constraint_3(X1, X2), [0], colors='b', linestyles='dashed')

# Plot the optimum solution
optimum_solution = (-1, -np.sqrt(2/3))
plt.scatter(optimum_solution[0], optimum_solution[1], color='red', label='Optimum Solution')
plt.text(optimum_solution[0], optimum_solution[1], 'Optimum Solution', ha='right')

# Set labels and title
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Minimization Problem Visualization')
plt.legend()

plt.show()
