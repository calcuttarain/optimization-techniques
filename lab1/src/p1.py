import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt

# a)
img_orig = plt.imread("../images/original.png")[:,:,0]
img_corr = plt.imread("../images/altered.png")[:,:,0]
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
fig.savefig("../images/denoised_image_a.png", dpi=300, bbox_inches='tight')


# b)
U = cp.Variable((rows, cols))

rho = 0.2

err = 0.5 * cp.sum_squares(U - img_corr)
vertical = cp.sum_squares(U[1:, :] - U[:-1, :])
orizontal = cp.sum_squares(U[:, 1:] - U[:, :-1])
f = err + rho * (vertical + orizontal)

obj = cp.Minimize(f)

prob = cp.Problem(obj)

result = prob.solve()
print("Optimal objective value: {:.4f}".format(prob.value))

img_denoised = U.value

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(img_corr, cmap='gray')
plt.title("Imaginea zgomotoasă Y")
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(img_denoised, cmap='gray')
plt.title("Imaginea reconstruită U*")
plt.axis('off')
plt.savefig("../images/denoised_image_b.png", dpi=300, bbox_inches='tight')
