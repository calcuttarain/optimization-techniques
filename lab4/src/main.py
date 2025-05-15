import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp

MIU = 1e10
np.random.seed(24)

def generate_separable_points(m, n, delta):
    w = np.random.randn(n)
    b = np.random.randn()

    norm_w = np.linalg.norm(w)

    points, y = [], []

    while len(points) < m:
        x = np.random.randn(n)
        margin = w @ x + b

        if margin >= delta * norm_w:
            points.append(x)
            y.append(1)
        elif margin <= -delta * norm_w:
            points.append(x)
            y.append(-1)

    return points, y, w, b

def generate_inseparable_points(m, n, epsilon):
    w = np.random.randn(n)
    b = np.random.randn()

    points, y = [], []

    for _ in range(m):
        x = np.random.randn(n)
        margin = w @ x + b

        if margin > epsilon:
            y.append(1)
        elif margin < -epsilon:
            y.append(-1)
        else:
            y.append(np.random.choice([-1, 1]))

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

'''
Introduc produsul scalar in problema duala pentru SVM inseparabil ca o constanta mare ori o functie de g(x).
Aleg acea functia patratica pentru ca e derivabila in 0 (spre deosebire de functia modul, spre exemplu).
Astfel, avem o functie de penalizare care se asigura ca produsul scalar este 0.
'''
def f_svm_dual(Q, y, my_lambda):
    sum_lambda = np.sum(my_lambda)

    Q_lambda = Q @ my_lambda
    Q_lambda_squared_norm = np.dot(Q_lambda, Q_lambda)

    return - 1 / 2 * Q_lambda_squared_norm + sum_lambda - MIU * (y @ my_lambda) ** 2

def gradient_svm_dual(Q, y, my_lambda):
    return - Q.T @ Q @ my_lambda + np.ones(len(my_lambda)) - 2 * MIU * (y @ my_lambda) * y 

def projection(my_lambda, rho):
    return np.clip(my_lambda, 0, rho)

def mgd_svm(Q, y, rho, c = .5, p = .5, numiter = 10000):
    my_lambda = np.random.uniform(low = 0, high = rho, size = len(y))
    my_lambda = projection(my_lambda, rho)

    for _ in range (numiter):
        gradient = gradient_svm_dual(Q, y, my_lambda)

        alpha_k = 1.0
        my_lambda_next = projection(my_lambda + alpha_k * gradient, rho)
        stop = 0

        # armijo
        while f_svm_dual(Q, y, my_lambda_next) <= f_svm_dual(Q, y, my_lambda) + c * alpha_k * gradient @ (my_lambda_next - my_lambda):
            alpha_k *= p 
            my_lambda_next = projection(my_lambda + alpha_k * gradient, rho)

            if stop == 0:
                break 
            stop -= 1

        my_lambda = my_lambda_next

    return my_lambda

def compute_solution(Q, X, y, my_lambda):
    # w_star scris matriceal
    w_star = Q @ my_lambda

    b_star_list = []

    support_vectors = np.where(my_lambda > 0)[0]
    for i in support_vectors:
        b_star_list.append(y[i] - w_star @ X[i, :])

    return w_star, b_star_list

def myplot(X, y, w_hat, b_hat, filename, title, hypr = True):
    X = np.array(X)

    d = len(X[0])

    fig = plt.figure(figsize=(16, 9))

    if d == 2:
        ax = fig.add_subplot(111)
        for i in range (len(X)):
            color = 'blue' if y[i] == -1 else 'red'
            ax.scatter(X[i][0], X[i][1], color=color)

        if hypr is True:
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

        if hypr is True:
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


# 1)
m, n, delta = 1000, 2, .3
X, y, w, b = generate_separable_points(m, n, delta)
w_hat, b_hat = solve_separable_svm(X, y, m, n)
myplot(X, y, w_hat.value, b_hat.value, 'separable_svm_2d', 'Separable SVM')


m, n, delta = 300, 3, .3
X, y, w, b = generate_separable_points(m, n, delta)
w_hat, b_hat = solve_separable_svm(X, y, m, n)
myplot(X, y, w_hat.value, b_hat.value, 'separable_svm_3d', 'Separable SVM')


# 2)
m, n, epsilon = 30, 2, 2.5
X, y, w, b = generate_inseparable_points(m, n, epsilon)
myplot(X, y, w, b, 'inseparable_data_2d', 'Inseparable SVM', hypr = False)

X = np.array(X)
y = np.array(y)

Q = np.array([y[i] * X[i, :] for i in range(len(y))]).T

rho = .2

my_lambda_star = mgd_svm(Q, y, rho)
w_star, b_star_list = compute_solution(Q, X, y, my_lambda_star)
myplot(X, y, w_star, np.mean(b_star_list), 'solved_inseparable_svm_2d', 'Solved Inseparable SVM')
