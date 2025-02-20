# Nolan Blevins

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar

# the objective function
def f(x):
    return 5*x[0]**2 + x[1]**2 + 2*x[2]**2 + 4*x[0]*x[1] - 14*x[0] - 6*x[1] + 20

# grad for obj func
def grad_f(x):
    df_dx1 = 10*x[0] + 4*x[1] - 14
    df_dx2 = 2*x[1] + 4*x[0] - 6
    df_dx3 = 4*x[2]
    return np.array([df_dx1, df_dx2, df_dx3])

# hessian mat for obj
def hessian_f(x):
    return np.array([[10, 4, 0], [4, 2, 0], [0, 0, 4]])

# steepest desc with bisec method
def steepest_descent(f, grad_f, x0, tol=1e-6, max_iter=1000):
    x = x0
    fs = [f(x)]
    xs = [x]
    for i in range(max_iter):
        gradient = grad_f(x)
        # neg grad for step desc
        direction = -gradient
        
        # bisection method for step size
        def phi(alpha): return f(x + alpha*direction)
        res = minimize_scalar(phi, bounds=(0, 1), method='bounded')
        step_size = res.x
        
        x = x + step_size * direction
        fs.append(f(x))
        xs.append(x)
        if np.linalg.norm(gradient) < tol:
            break
    return x, fs, xs

# newton method
def newtons_method(f, grad_f, hessian_f, x0, tol=1e-6, max_iter=100):
    x = x0
    fs = [f(x)]
    xs = [x]
    for i in range(max_iter):
        gradient = grad_f(x)
        H = hessian_f(x)
        direction = -np.linalg.solve(H, gradient)
        # line search for step size ( full step )
        step_size = 1
        x = x + step_size * direction
        fs.append(f(x))
        xs.append(x)
        if np.linalg.norm(gradient) < tol:
            break
    return x, fs, xs

# initial guess at origin
x0 = np.array([0.0, 0.0, 0.0])

# run steepest descent
x_min_sd, fs_sd, xs_sd = steepest_descent(f, grad_f, x0)

# run newton method
x_min_nr, fs_nr, xs_nr = newtons_method(f, grad_f, hessian_f, x0)

# Plotting
fig, axs = plt.subplots(1, 2, figsize=(14, 6))

# Cost vs Iteration
axs[0].plot(fs_sd, label='Steepest Descent')
axs[0].plot(fs_nr, label='Newton\'s Method')
axs[0].set_title('Cost vs Iteration')
axs[0].set_xlabel('Iteration')
axs[0].set_ylabel('Cost')
axs[0].legend()

# Prepare contour plot
X1, X2 = np.meshgrid(np.linspace(-5, 5, 400), np.linspace(-5, 5, 400))
Z = np.array([f([x1, x2, 0]) for x1, x2 in zip(np.ravel(X1), np.ravel(X2))])
Z = Z.reshape(X1.shape)

# Cost contours and evolution of the solution sequence
axs[1].contour(X1, X2, Z, levels=np.logspace(-2, 3, 20), cmap='viridis')
axs[1].plot(*zip(*[(x[0], x[1]) for x in xs_sd]), 'o-', label='Steepest Descent Path')
axs[1].plot(*zip(*[(x[0], x[1]) for x in xs_nr]), 'o-', label='Newton\'s Method Path')
axs[1].set_title('Cost Contours and Solution Sequence')
axs[1].set_xlabel('$x_1$')
axs[1].set_ylabel('$x_2$')
axs[1].legend()

plt.tight_layout()
plt.show()
