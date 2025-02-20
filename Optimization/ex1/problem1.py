import numpy as np
import matplotlib.pyplot as plt

# objective func
def f(x):
    return 5*x[0]**2 + x[1]**2 + 2*x[2]**2 + 4*x[0]*x[1] - 14*x[0] - 6*x[1] + 20

# gradient of the objective function (found by hand)
def grad_f(x):
    return np.array([10*x[0] + 4*x[1] - 14, 4*x[0] + 2*x[1] - 6, 4*x[2]])

# line search for minimization rule
# iteratively reduces alpha until improvment is met
# https://machinelearningmastery.com/line-search-optimization-with-python/
def line_search(f, grad_f, x, direction, alpha=1, beta=0.5, sigma=0.1):
    while f(x + alpha * direction) > f(x) + sigma * alpha * np.dot(grad_f(x), direction):
        alpha *= beta
    return alpha

# Steepest Descent Algorithm
def steepest_descent(f, grad_f, initial_guess, step_size_strategy, max_iter=100, tol=1e-6):
    x = np.array(initial_guess) # init intial guess
    costs = [f(x)] # record intial cost
    
    for i in range(max_iter):
        gradient = grad_f(x)
        if np.linalg.norm(gradient) < tol:  # check if norm is below tolerance threshold 
            break
        
        direction = -gradient  # steepest descent direction
        
        if step_size_strategy == "constant":
            alpha = 0.01  # constant step size
        elif step_size_strategy == "diminishing":
            alpha = 1 / (i + 1)  # diminishing step size
        elif step_size_strategy == "minimization":
            alpha = line_search(f, grad_f, x, direction) # uses line search method to select optimal alpha
        
        # step size chosen based on current method
        x = x + alpha * direction
        # update x value
        costs.append(f(x))
    
    return x, costs

# plotting function
def plot_costs(costs_dict):
    plt.figure(figsize=(10, 6))
    for strategy, costs in costs_dict.items():
        plt.plot(costs, label=f'{strategy} step size')
    plt.xlabel('Iteration')
    plt.ylabel('Cost')
    plt.title('Cost vs Iteration for Different Step Size Strategies')
    plt.legend()
    plt.grid(True)
    plt.show()

# running SD with different strategies
initial_guess = [0, 0, 0]
strategies = ["constant", "diminishing", "minimization"]
results = {}

for strategy in strategies:
    minimizer, costs = steepest_descent(f, grad_f, initial_guess, strategy)
    results[strategy] = costs

# print minimum costs
for strategy, costs in results.items():
    min_cost = min(costs)
    print(f"Minimum cost for {strategy} step size strategy: {min_cost:.6f}")

# plotting the results
plot_costs(results)
