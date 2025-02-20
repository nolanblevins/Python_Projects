import numpy as np
import matplotlib.pyplot as plt

# Define the function
def f(x):
    return 3*x**4 - 4*x**3 + 1

# Define the derivative and second derivative
def f_prime(x):
    return 12*x**3 - 12*x**2

def f_double_prime(x):
    return 36*x**2 - 24*x

# Find critical points where f''(x) = 0
inflection_points = np.roots([36, -24])

# Generate x values
x_values = np.linspace(-10, 10, 400)
y_values = f(x_values)

# Increase figure size
plt.figure(figsize=(12, 8))

# Plot the function
plt.plot(x_values, y_values, label='$f(x) = 3x^4 - 4x^3 + 1$')

# Mark the point of inflection
plt.scatter(0, 1, color='red', label='Point of Inflection', zorder=5)

# Set y-axis limits
plt.ylim(0, 5)

# Add labels and title
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Graph of $f(x) = 3x^4 - 4x^3 + 1$ with Point of Inflection')
plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)
plt.grid(color='gray', linestyle='--', linewidth=0.5)

# Add legend
plt.legend()

plt.show()
