import numpy as np
import matplotlib.pyplot as plt

# Define the constraints
def ellipse_constraint(x1):
    return np.sqrt((36 - 4*x1**2) / 9)

def curve_constraint(x1):
    return (3 - x1**2) / 3

# Generate x1 values
x1_values = np.linspace(-10, 10, 400)

# Plot the constraints
plt.plot(x1_values, ellipse_constraint(x1_values), label='Ellipse Constraint')
plt.plot(x1_values, curve_constraint(x1_values), label='Curve Constraint')
plt.fill_between(x1_values, np.minimum(ellipse_constraint(x1_values), curve_constraint(x1_values)), -3, color='gray', alpha=0.5, label='Feasible Region')

# Plot the optimum solution
plt.scatter(3, 1, color='red', marker='*', label='Optimum Solution')

# Add labels and legend
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.title('Feasible Region and Optimum Solution')
plt.legend()

# Set plot limits and display
plt.xlim(-10, 10)
plt.ylim(-10, 10)
plt.grid(True)
plt.gca().set_aspect('equal', adjustable='box')
plt.show()
