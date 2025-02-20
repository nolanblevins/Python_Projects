import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

# Objective function
def objective_function(x, y, a, b):
    return (x - a)**2 + (y - b)**2

# Constraints
def constraint_1(x, y):
    return (x - 8)**2 + (y - 9)**2 - 49

def constraint_2(x, y):
    return x + y - 24

# Define the range of x and y values
x_values = np.linspace(0, 25, 100)
y_values = np.linspace(0, 25, 100)

# Create a meshgrid
x_mesh, y_mesh = np.meshgrid(x_values, y_values)

# Define constants
a = 16
b = 14

# Evaluate the objective function and constraints on the meshgrid
objective_values = objective_function(x_mesh, y_mesh, a, b)
constraint1_values = constraint_1(x_mesh, y_mesh)
constraint2_values = constraint_2(x_mesh, y_mesh)

# Plot the objective function
plt.figure(figsize=(10, 8))
contour = plt.contour(x_mesh, y_mesh, objective_values, levels=6, colors='k', label='Objective Function')
plt.clabel(contour, inline=True, fontsize=8)

# Plot the constraints
plt.contour(x_mesh, y_mesh, constraint1_values, levels=[0], colors='r', linestyles='solid', label='Constraint 1')
plt.contour(x_mesh, y_mesh, constraint2_values, levels=[0], colors='b', linestyles='dashed', label='Constraint 2')

# Add labels and legends
plt.xlabel('x')
plt.ylabel('y')
plt.title('Optimization Problem Example 7')
plt.legend()

# Add a legend for the constraint circles
circle = Circle((8, 9), 7, fill=False, color='r', linestyle='dashed', label='Feasible Region')
plt.gca().add_patch(circle)

# Highlight the feasible region
#plt.fill_between(x_values, 3, np.minimum(24-x_values, 13), where=(x_values >= 2) & (x_values <= 13), color='gray', alpha=0.3)
# Add vertical lines at x = 2 and x = 13
plt.axvline(x=2, color='g', linestyle='-', label='x = 2')
plt.axvline(x=13, color='g', linestyle='-', label='x = 13')
plt.show()
