# Nolan Blevins

import numpy as np
import matplotlib.pyplot as plt

# given data set
t = np.array([-2, -1, 0, 1, 2])
y = np.array([2, -10, 0, 2, 1])

# objective function
def objective_function(x, t, y):
    return 0.5 * np.sum((y - (x[0] + x[1] * t + x[2] * t**2))**2)

# grad of obj func
def gradient(x, t, y):
    grad = np.zeros(3)
    grad[0] = np.sum(x[0] + x[1] * t + x[2] * t**2 - y)
    grad[1] = np.sum((x[0] + x[1] * t + x[2] * t**2 - y) * t)
    grad[2] = np.sum((x[0] + x[1] * t + x[2] * t**2 - y) * t**2)
    return grad

# hesssian of obj func
def hessian(x, t, y):
    H = np.zeros((3, 3))
    H[0, 0] = len(t)
    H[0, 1] = H[1, 0] = np.sum(t)
    H[0, 2] = H[2, 0] = np.sum(t**2)
    H[1, 1] = np.sum(t**2)
    H[1, 2] = H[2, 1] = np.sum(t**3)
    H[2, 2] = np.sum(t**4)
    return H


# sd implementation
def steepest_descent(x0, t, y, alpha, tol=1e-6, max_iter=1000):
    x = x0.copy()
    for i in range(max_iter):
        grad = gradient(x, t, y)
        x_new = x - alpha * grad
        if np.linalg.norm(x_new - x) < tol:
            break
        x = x_new
    return x


# newtons meth implementation
def newtons_method(x0, t, y, tol=1e-6, max_iter=1000):
    x = x0.copy()
    for i in range(max_iter):
        grad = gradient(x, t, y)
        H = hessian(x, t, y)
        x_new = x - np.linalg.inv(H) @ grad
        if np.linalg.norm(x_new - x) < tol:
            break
        x = x_new
    return x

# initial guess at origin
x0 = np.array([0, 0, 0])


# run sd and newtons
coefficients_sd = steepest_descent(x0, t, y, alpha=0.001)
coefficients_nr = newtons_method(x0, t, y)

# t plot for fietted
t_plot = np.linspace(-2.5, 2.5, 100)


# plot fitted polynomials
plt.figure(figsize=(12, 8))
y_fitted_sd = np.polyval(coefficients_sd[::-1], t_plot)
plt.plot(t_plot, y_fitted_sd, label='Fitted Polynomial (SD)', color='green', linestyle='--')
y_fitted_nr = np.polyval(coefficients_nr[::-1], t_plot)
plt.plot(t_plot, y_fitted_nr, label='Fitted Polynomial (NR)', color='magenta', linestyle='-.')
plt.scatter(t, y, color='red', label='Original Data Points')
plt.xlabel('t')
plt.ylabel('y')
plt.title('Comparison of Fitted Polynomials and Original Data')
plt.legend()
plt.grid(True)
plt.show()
