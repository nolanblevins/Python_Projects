# Nolan Blevins

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# given data set
X = np.array([-1, -0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
Y = np.array([-0.96, -0.577, -0.073, 0.377, 0.641, 0.66, 0.461, 0.134, -0.201, -0.434, -0.5, -0.393, -0.165, 0.099, 0.307, 0.396, 0.345, 0.182, -0.031, -0.219, -0.321])

# function def
def f(X, params):
    x0, x1, x2, x3, x4 = params
    return x0 + x1 * np.sin(x2 * X) + x3 * np.cos(x4 * X)

# cost as sum of sq
def cost_function(params, X, Y):
    return np.sum((Y - f(X, params))**2)

# inital guess
params_initial = np.array([0, 1, 1, 1, 1])

# minimize to find optimal params
result = minimize(cost_function, params_initial, args=(X, Y), method='BFGS')

# extract the optimized parameters
params_optimized = result.x

# generate Y values using the optimized parameters
Y_fitted = f(X, params_optimized)

# plot fitted and original
plt.figure(figsize=(14, 7))

# original data
plt.scatter(X, Y, color='red', label='Original Data')

# fitted function
plt.plot(X, Y_fitted, label='Fitted Function', color='blue')

plt.title('Fitted Function vs Original Data')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.grid(True)
plt.show()

# plot cost vs iteration
plt.figure(figsize=(14, 7))
plt.plot(result.jac, label='Cost vs Iteration', color='green')
plt.title('Cost vs Iteration')
plt.xlabel('Iteration')
plt.ylabel('Cost')
plt.legend()
plt.grid(True)
plt.show()

# return the optimized parameters and final cost for reference
params_optimized, result.fun
