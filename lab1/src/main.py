import scipy.optimize as optimize
import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt

# 1.1
x0 = (5,5)
A = np.ones((2,2)) + np.eye(2)
fun = lambda x : x.T @ A @ x + x[0] * x[1]
res = optimize.minimize(fun,x0,method='Nelder-Mead')
# print(res.x)


# 1.2
def f(x):
    return np.sqrt((x[0] - 3) ** 2 + (x[1] - 2) **2 )

def constraint(x):
    return np.atleast_1d(1.5 - np.sum(np.abs(x)))

res = optimize.minimize(f, np.array([0, 0]), method="SLSQP", constraints={"fun": constraint, "type": "ineq"})

# print(res)


# 2.1
x = cp.Variable()
y = cp.Variable()

constraints = [x + y == 1, 
               x - y >= 1]

obj = cp.Minimize((x - y) ** 2)

prob = cp.Problem(obj, constraints)
prob.solve()

# print(f'Status: {prob.status}')
# print(f'Optimal value: {prob.value}')
# print(f'Optimal var: {x.value}, {y.value}')


# 3.1
img_orig = plt.imread("../images/original.jpeg")[:,:,0]
img_corr = plt.imread("../images/altered.jpeg")[:,:,0]
rows, cols = img_orig.shape

known = np.zeros((rows, cols))
for i in range(rows):
    for j in range(cols):
        if img_corr[i, j] == img_orig[i, j]:
            known[i, j] = 1

fig, ax = plt.subplots(1, 3, figsize=(10, 5))
ax[0].imshow(img_orig, cmap='gray')
ax[0].set_title("Original Image")
ax[0].axis('off')
ax[1].imshow(img_corr, cmap='gray')
ax[1].set_title("Corrupted Image")
ax[1].axis('off')

U = cp.Variable(shape=(rows, cols))
obj = cp.Minimize(cp.tv(U))
constraints = [cp.multiply(known, U) == cp.multiply(known, img_corr)]
prob = cp.Problem(obj, constraints)

prob.solve(verbose=True, solver=cp.SCS)
print("optimal objective value: {}".format(obj.value))

img_denoised = U.value
if U.value is None:
    raise ValueError("Optimization failed! U.value is None.")

img_denoised = np.array(U.value, dtype=np.float64)
img_denoised[known == 1] = img_orig[known == 1]

ax[2].imshow(img_denoised, cmap='gray')
ax[2].set_title("Denoised Image")
ax[2].axis('off')
fig.savefig("../images/denoised_image.png", dpi=300, bbox_inches='tight')

