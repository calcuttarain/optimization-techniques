import numpy as np
import scipy.optimize as optimize
import cvxpy as cp

np.random.seed(10)

n = 8
m = 5

M = np.random.randn(n, n)
# matrice pozitiv semidefinita
Q = M.T @ M

A = np.random.randn(m, n)

c = np.random.randn(n)

# iau un b valid
sol = np.random.rand(n)
b = A @ sol

# scipy.optimize
def f(x):
    return 0.5 * np.dot(x, Q @ x) + np.dot(c, x)

def gradient(x):
    return Q @ x + c

def constraint(x):
    return A @ x - b

x0 = np.zeros(n)
res = optimize.minimize(f, x0, method="SLSQP", jac = gradient, constraints={"fun": constraint, "type": "eq"}, bounds=[(0, None)] * n)

print("Rezolvare cu Scipy.optimize:")
print("Solutia optima x* =", res.x)
print("Valoarea obiectivului:", res.fun)


# cvxpy
x = cp.Variable(n)

objective_cvx = 0.5 * cp.quad_form(x, Q) + c @ x

constraints_cvx = [A @ x == b, x >= 0]

prob = cp.Problem(cp.Minimize(objective_cvx), constraints_cvx)
result = prob.solve()

print("\nRezolvare cu CVXPY:")
print("Solutia optima x* =", x.value)
print("Valoarea obiectivului:", prob.value)
