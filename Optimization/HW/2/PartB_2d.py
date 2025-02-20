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

# Find the Minkowski sum
X1_sum, X2_sum, X3_sum = np.meshgrid(x1, x2, x3)
X1_sum = X1_sum.flatten()
X2_sum = X2_sum.flatten()
X3_sum = X3_sum.flatten()

# Plot the Minkowski sum
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the Minkowski sum points
ax.scatter(X1_sum, X2_sum, X3_sum, color='b', alpha=0.3)

# Set labels and title
ax.set_xlabel('X1')
ax.set_ylabel('X2')
ax.set_zlabel('X3')
ax.set_title('Visualization of S1 ⊕ S2')

plt.show()
