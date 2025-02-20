# Nolan Blevins

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# function to apply the Gram-Schmidt
def gram_schmidt(A):
    m, n = A.shape
    Q = np.zeros((m, n))
    R = np.zeros((n, n))
    
    for k in range(n):
        Q[:, k] = A[:, k]
        for i in range(k):
            R[i, k] = np.dot(Q[:, i].T, A[:, k])
            Q[:, k] = Q[:, k] - R[i, k]*Q[:, i]
        R[k, k] = np.linalg.norm(Q[:, k])
        if R[k, k] == 0: # check to prevent division by zero
            Q[:, k] = np.zeros(m) # assign zero vector in case of linear dependency
        else:
            Q[:, k] = Q[:, k] / R[k, k]
    
    return Q, R

# given column vectors
vectors = np.array([
    [1, -1, 0],
    [1, 1, 2]
])

# applying the Gram-Schmidt to vectors
Q_independent, _ = gram_schmidt(vectors.T)

# visualization
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# plotting the original vectors
ax.quiver(0, 0, 0, vectors[0, 0], vectors[0, 1], vectors[0, 2], color='r', label='Original Vector 1')
ax.quiver(0, 0, 0, vectors[1, 0], vectors[1, 1], vectors[1, 2], color='b', label='Original Vector 2')

# plotting the orthonormal vectors
ax.quiver(0, 0, 0, Q_independent[0, 0], Q_independent[1, 0], Q_independent[2, 0], color='g', label='Orthonormal Vector 1', linestyle='dotted')
ax.quiver(0, 0, 0, Q_independent[0, 1], Q_independent[1, 1], Q_independent[2, 1], color='y', label='Orthonormal Vector 2', linestyle='dotted')

# setting the plot dimensions and labels
ax.set_xlim([-1, 1])
ax.set_ylim([-1, 1])
ax.set_zlim([-1, 2])
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Z axis')
ax.legend()

plt.title('Original and Orthonormal Vectors')
plt.show()

