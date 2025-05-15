import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp
import scipy

np.random.seed(22)

def generate_separable_points(m, n):
    w = np.random.randn(n)
    b = np.random.randn()

    points, y = [], []

    for _ in range (m):
        x = np.random.randn(n)

        y.append(1 if w @ x + b > 0 else -1)

        points.append(x)

    return points, y, w, b

def solve_separable_svm(X, y, m, n):
    w_hat = cp.Variable(n)
    b_hat = cp.Variable()

    objective = cp.Minimize(cp.sum_squares(w_hat))

    constraints = []
    for i in range (m):
        constraints.append(y[i] * (X[i] @ w_hat + b_hat) >= 1)

    problem = cp.Problem(objective, constraints)

    problem.solve()

    return w_hat, b_hat

def myplot(X, y, w_hat, b_hat, filename, title):
    X = np.array(X)

    d = len(X[0])

    fig = plt.figure(figsize=(16, 9))

    if d == 2:
        ax = fig.add_subplot(111)
        for i in range (len(X)):
            color = 'blue' if y[i] == -1 else 'red'
            ax.scatter(X[i][0], X[i][1], color=color)

        x_vals = np.linspace(np.min(X[:, 0]) - 1, np.max(X[:, 0]) + 1, 100)
        slope = -w_hat[0] / w_hat[1]
        intercept = -b_hat / w_hat[1]
        y_vals = slope * x_vals + intercept
        plt.plot(x_vals, y_vals, 'k--', label='hiperplan')

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.grid(True)
        ax.legend()

    elif d == 3:
        ax = fig.add_subplot(111, projection='3d')
        for i in range (len(X)):
            color = 'blue' if y[i] == -1 else 'red'
            ax.scatter(X[i][0], X[i][1], X[i][2], color=color)

        x_range = np.linspace(np.min(X[:, 0]), np.max(X[:, 0]), 20)
        y_range = np.linspace(np.min(X[:, 1]), np.max(X[:, 1]), 20)
        x_grid, y_grid = np.meshgrid(x_range, y_range)

        w1, w2, w3 = w_hat
        z_grid = (-w1 * x_grid - w2 * y_grid - b_hat) / w3
        ax.plot_surface(x_grid, y_grid, z_grid, alpha=0.3, color='green', label='hiperplan')

        ax.legend()

    plt.title(title)
    plt.savefig(f'../plots/{filename}.png', dpi = 300)
    plt.clf()


m, n = 100, 2
X, y, w, b = generate_separable_points(m, n)
w_hat, b_hat = solve_separable_svm(X, y, m, n)
myplot(X, y, w_hat.value, b_hat.value, 'separable_svm_2d', 'Separable SVM')


m, n = 100, 3
X, y, w, b = generate_separable_points(m, n)
w_hat, b_hat = solve_separable_svm(X, y, m, n)
myplot(X, y, w_hat.value, b_hat.value, 'separable_svm_3d', 'Separable SVM')
