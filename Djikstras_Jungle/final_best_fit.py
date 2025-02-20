# nolan blevins
import numpy as np

solutions = {
    '(x1, x2)': np.array([
        [0.8503, 0.7438],  # LS
        [0.8113, 0.7086],  # 2nd
        [0.7273, 0.6364]   # Simplex
    ]),
    '(x1, x3)': np.array([
        [0.8503, 1.2548],  # LS
        [0.8113, 1.2469],  # 2nd
        [0.7273, 1.0000]   # Simplex
    ]),
    '(x2, x3)': np.array([
        [0.7438, 1.2548],  # LS
        [0.7086, 1.2469],  # 2nd
        [0.6364, 1.0000]   # Simplex
    ])
}

targets = {
    '(x1, x2)': 2,  
    '(x1, x3)': 2,  
    '(x2, x3)': 2   
}

best_fit = {}
for subspace, subspace_solutions in solutions.items():
    residuals = []
    for solution in subspace_solutions:
        residual = abs(np.sum(solution) - targets[subspace])
        residuals.append(residual)
    best_method_index = np.argmin(residuals)
    best_fit[subspace] = (['LS', '2nd', 'Simplex'][best_method_index], residuals[best_method_index])

# Output the results
for subspace, result in best_fit.items():
    print(f"Best fit in {subspace}: {result[0]} with residual {result[1]}")
