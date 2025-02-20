import numpy as np
import matplotlib.pyplot as plt

# used as reference for least squares
# https://pythonnumericalmethods.berkeley.edu/notebooks/chapter16.04-Least-Squares-Regression-in-Python.html
# https://vnav.mit.edu/material/17-18-NonLinearLeastSquares-notes.pdf
# https://www.mathworks.com/help/optim/ug/least-squares-model-fitting-algorithms.html


# input
t = np.array([-2, -1, 0, 1, 2])
# output 
y = np.array([2, -10, 0, 2, 1])

# construct matrix A and vector b
# np.vstack used to stack arrays in sequence vertically
A = np.vstack([np.ones_like(t), t, t**2]).T
b = y

# solves equation of A^T Ax = A^T b
x = np.linalg.solve(A.T @ A, A.T @ b)

# display the found coefficients
print("Coefficients of the fitted polynomial are:", x)

# generating points for the fitted polynomial
t_fine = np.linspace(-2.5, 2.5, 400)
p_t = x[0] + x[1]*t_fine + x[2]*t_fine**2

# plotting
plt.figure(figsize=(8, 5))
plt.plot(t_fine, p_t, label=f'Fitted Polynomial $p(t) = {x[0]} + {x[1]}t + {x[2]}t^2$', color='blue')
plt.scatter(t, y, color='red', label='Given Data')
plt.title('Fitted Polynomial and Given Data')
plt.xlabel('t')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()
