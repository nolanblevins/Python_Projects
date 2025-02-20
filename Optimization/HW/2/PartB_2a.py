import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Generate points for the plane
x = np.linspace(-10, 10, 100)
y = x
X, Y = np.meshgrid(x, y)
Z = X

# Plot the plane
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, alpha=0.5)

# Set labels and title
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Visualization of S1: {x∈R^3 | x1 = x2, x2 = x3}')

plt.show()
