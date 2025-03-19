import scipy.optimize as optimize
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

def f(x):
    return 100 * (x[1] - x[0] ** 2) ** 2 + (1 - x[0]) ** 2

result = optimize.minimize(f, np.array([1, 0]), method="COBYLA")
print(f'Optim: {result.x}')
print(f'Valoare optim: {result.fun}')

x1 = np.linspace(-2, 2, 400)
x2 = np.linspace(-1, 3, 400)
X1, X2 = np.meshgrid(x1, x2)
Z = 100 * (X2 - X1 ** 2) ** 2 + (1 - X1) ** 2

plt.figure(figsize=(16, 9))
contours = plt.contour(X1, X2, Z, levels=np.logspace(-0.5, 3.5, 20), norm=LogNorm(), cmap='jet')

plt.plot(1, 1, 'ko', markersize=15, label='(1,1)')
plt.plot(result.x[0], result.x[1], 'ro', markersize=8, label='Optim gasit')

plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.legend()
plt.savefig("../plots/pb2.png", dpi=300, bbox_inches='tight')
