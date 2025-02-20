import numpy as np
import matplotlib.pyplot as plt

# constraints
def constraint1(x1, x2):
    return 4*x1**2 + 9*x2**4 - 36

def constraint2(x1, x2):
    return x1**2 + 3*x2**2 - 3

# x1 values
x1 = np.linspace(-1, 3, 400)
# x2 values
x2 = np.linspace(-2, 2, 400)

X1, X2 = np.meshgrid(x1, x2)
C1 = constraint1(X1, X2)
C2 = constraint2(X1, X2)

# figure size
plt.figure(figsize=(8, 6))

# plot for the constraints

# constraint 1
plt.contour(X1, X2, C1, levels=[0], colors='red', linewidths=2, linestyles='dashed')
# constraint 2
plt.contour(X1, X2, C2, levels=[0], colors='blue', linewidths=2, linestyles='dotted')

# fill feasible region
plt.contourf(X1, X2, C1, levels=[-np.inf, 0], colors='red', alpha=0.2)
plt.contourf(X1, X2, C2, levels=[0, np.inf], colors='blue', alpha=0.2)

plt.xlim([-1, 3])
plt.ylim([-2, 2])
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.title('Problem 4 feasible region')

# show plot
plt.grid(True)
plt.axhline(0, color='black',linewidth=0.5)
plt.axvline(0, color='black',linewidth=0.5)
plt.show()
