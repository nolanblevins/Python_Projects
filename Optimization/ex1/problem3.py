import numpy as np
import matplotlib.pyplot as plt

# x points
X = np.array([-1, -0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
# y points
Y = np.array([-0.96, -0.577, -0.073, 0.377, 0.641, 0.66, 0.461, 0.134, -0.201, -0.434, -0.5, -0.393, -0.165, 0.099, 0.307, 0.396, 0.345, 0.182, -0.031, -0.219, -0.321])

# funct to find val of function at specific points
def f(X, x0, x1, x2, x3):
    return x0 + x1 * (np.sin(X) + np.cos(X)) + x2 * (np.sin(2*X) + np.cos(2*X)) + x3 * (np.sin(3*X) + np.cos(3*X))

# cost function
# computes mean squared error between
# prefdicted outputs and actual outputs
def cost_function(X, Y, x0, x1, x2, x3):
    return np.mean((f(X, x0, x1, x2, x3) - Y) ** 2)

# calculates gradient with respect to parameters
def gradient(X, Y, x0, x1, x2, x3):
    n = len(X)
    f_x = f(X, x0, x1, x2, x3)
    grad_x0 = 2/n * np.sum(f_x - Y)
    grad_x1 = 2/n * np.sum((f_x - Y) * (np.sin(X) + np.cos(X)))
    grad_x2 = 2/n * np.sum((f_x - Y) * (np.sin(2*X) + np.cos(2*X)))
    grad_x3 = 2/n * np.sum((f_x - Y) * (np.sin(3*X) + np.cos(3*X)))
    return np.array([grad_x0, grad_x1, grad_x2, grad_x3])

# grad descent
# itertive approach where each iteration
# calculates the value of each param
# in negative descent direction times alpha
# also monitors convergence
def gradient_descent(X, Y, x0, x1, x2, x3, alpha, iterations):
    cost_history = []
    for i in range(iterations):
        grad = gradient(X, Y, x0, x1, x2, x3)
        x0 -= alpha * grad[0]
        x1 -= alpha * grad[1]
        x2 -= alpha * grad[2]
        x3 -= alpha * grad[3]
        cost = cost_function(X, Y, x0, x1, x2, x3)
        cost_history.append(cost)
    return x0, x1, x2, x3, cost_history

# initial parameters
x0, x1, x2, x3 = 0, 0, 0, 0
alpha = 0.01
iterations = 1000

# run gradient descent
opt_x0, opt_x1, opt_x2, opt_x3, cost_history = gradient_descent(X, Y, x0, x1, x2, x3, alpha, iterations)
# print the minimum cost achieved
min_cost = min(cost_history)
print("Minimum cost achieved:", min_cost)
# plot fitted function and original data
plt.figure(figsize=(10, 5))
plt.scatter(X, Y, color='red', label='Data Points')
X_fit = np.linspace(-1, 1, 100)
Y_fit = f(X_fit, opt_x0, opt_x1, opt_x2, opt_x3)
plt.plot(X_fit, Y_fit, label='Fitted Function')
plt.legend()
plt.title('Fitted Function vs Original Data')
plt.xlabel('X')
plt.ylabel('Y')
plt.grid(True)
plt.show()

# plot cost vs iteration
plt.figure(figsize=(10, 5))
plt.plot(cost_history)
plt.title('Cost vs Iteration')
plt.xlabel('Iteration')
plt.ylabel('Cost')
plt.grid(True)
plt.show()

(opt_x0, opt_x1, opt_x2, opt_x3), cost_history[-1]
