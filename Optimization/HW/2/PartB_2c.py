import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Generate points for the plane of S1
x1 = np.linspace(-10, 10, 100)
x2 = x1
X1, X2 = np.meshgrid(x1, x2)
X3 = X2

# Generate points for the plane of S2
x3 = np.linspace(-10, 10, 100)
X3_s2, X1_s2 = np.meshgrid(x3, x1)
X2_s2 = 1 - X1_s2 - X3_s2

# Plot the intersection
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the intersection
ax.plot_surface(X1, X2, X3, color='b', alpha=0.3, label='S1')
ax.plot_surface(X1_s2, X2_s2, X3_s2, color='r', alpha=0.3, label='S2')

# Find the intersection
intersection_X1 = x1
intersection_X2 = intersection_X1
intersection_X3 = 1 - intersection_X1 - intersection_X2

# Plot the intersection points
ax.scatter(intersection_X1, intersection_X2, intersection_X3, color='g', label='Intersection Points')

# Set labels and title
ax.set_xlabel('X1')
ax.set_ylabel('X2')
ax.set_zlabel('X3')
ax.set_title('Visualization of S1 ∩ S2: {x∈R^3 | x1 = x2, x2 = x3, x1 + x2 + x3 = 1}')

plt.show()
