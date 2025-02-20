# Nolan Blevins
# Minimization Example 3
import matplotlib.pyplot as plt
import numpy as np

# Objective function
def objective_function(x1, x2):
    return -2 * x1 - x2

# Constraints
def constraint1(x):
    return (4 - x) / (8/3)

def constraint2(x):
    return 2 - x

def constraint3(x):
    return 3 / 2

# Generate x values
x_values = np.linspace(0, 3, 100)

# Plot objective function
plt.plot(x_values, objective_function(x_values, constraint1(x_values)), label='-2x1 - x2', color='blue')

# Plot constraints
plt.plot(x_values, constraint1(x_values), label='x1 + (8/3)x2 <= 4', color='green')
plt.plot(x_values, constraint2(x_values), label='x1 + x2 <= 2', color='orange')
plt.axvline(x=constraint3(x_values[0]), linestyle='--', label='2x1 <= 3', color='red')

# Highlight feasible region
plt.fill_between(x_values, 0, np.minimum(constraint1(x_values), constraint2(x_values)), where=(x_values >= 0) & (x_values <= 3/2), color='gray', alpha=0.3, label='Feasible Region')

# Plot the minimizer point
minimizer_x1, minimizer_x2 = 1.5, 0.5
plt.scatter(minimizer_x1, minimizer_x2, color='black', marker='o', label='Minimizer')

# Label the minimizer point
plt.text(minimizer_x1, minimizer_x2, '  Minimizer', verticalalignment='bottom', horizontalalignment='right', color='black')

# Set labels and legend
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Example 3 Minimization')
plt.legend()

# Set axis limits
plt.xlim(0, 3)
plt.ylim(0, 3)

# Show the plot
plt.show()
