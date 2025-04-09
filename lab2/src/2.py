import numpy as np
import matplotlib.pyplot as plt

np.random.seed(22)

# punctele fixe
def genereaza_sedii(m, d):
    x_min, x_max = -1, 1
    y_min, y_max = -1, 1
    z_min, z_max = -1, 1

    if d == 2:
        coordonate_sedii = np.random.uniform(low = [x_min, y_min], high = [x_max, y_max], size = (m, d))
    else: 
        coordonate_sedii = np.random.uniform(low = [x_min, y_min, z_min], high = [x_max, y_max, z_max], size = (m, d))

    return coordonate_sedii

# punctul de plecare pentru coordonatele depozitelor
def init_depozite(n, d):
    x_min, x_max = -1, 1
    y_min, y_max = -1, 1
    z_min, z_max = -1, 1

    if d == 2:
        coordonate_depozite = np.random.uniform(low = [x_min, y_min], high = [x_max, y_max], size = (n, d))
    else: 
        coordonate_depozite = np.random.uniform(low = [x_min, y_min, z_min], high = [x_max, y_max, z_max], size = (n, d))

    return coordonate_depozite

# o fac in asa fel incat sa nu existe conexiuni sediu - sediu, doar depozit - sediu, depozit - depozit
def genereaza_matrice_adiacenta(m, n, ponderata = False, cost_min = 0, cost_max = 50, connection_prob=0.7):
    total_points = m + n
    
    W = np.zeros((total_points, total_points), dtype=int)
    
    for i in range(m):  
        for j in range(m, total_points):  
            if ponderata:
                cost = np.random.uniform(cost_min, cost_max)
                W[i][j] = cost
                W[j][i] = cost
            elif np.random.rand() < connection_prob:
                W[i][j] = 1
                W[j][i] = 1
    
    return W


def f(coords, W, p):
    n = coords.shape[0]
    
    sigma = 0.0
    
    for i in range(n):
        for j in range(i + 1, n):
            if W[i][j]:  
                dist = np.linalg.norm(coords[i] - coords[j])
                sigma += (W[i][j] * dist) ** p
    
    return sigma

def gradient(coords, W, p, m):
    n = coords.shape[0]

    gradient_coords = np.zeros(coords.shape)
    
    for i in range(m, n):
        for j in range(n):
            if W[i][j]:  
                dist = np.linalg.norm(coords[i] - coords[j])
                gradient_coords[i] += W[i, j] * p * (dist ** (p-2)) * (coords[i] - coords[j])
    
    return gradient_coords

# gradient_descend cu backtracking (ptc nu e necesar sa aflu constanta Lipschitz sau sa minimizez functia de la steepest)
def gradient_descend(A, W, p, m, e = 1e-3, rho = 0.5, c = 0.5, max_iter = 100000):
    intermediars = []
    alphas = []

    for i in range (max_iter):
        grad_x = gradient(A, W, p, m)
        if np.linalg.norm(grad_x) < e:
            print(f"minim gasit dupa {i} iteratii")
            break

        alpha_k = 1.0
        t = 0
        while f(A - alpha_k * grad_x, W, p) > (f(A, W, p) - c * alpha_k * np.linalg.norm(grad_x) ** 2):
            alpha_k = rho * alpha_k
            if t > 100:
                break
            t += 1

        A -= alpha_k * grad_x

        alphas.append(alpha_k)
        intermediars.append(f(A, W, p))

    print(f"f* = {f(A, W, p)}")
    print(f"p = {p}")
    return A, intermediars, alphas


# def stochastic_gradient_descend(A, W, p, m, N, beta = 0.4, T = 3000, comparison = False):
#     alpha = 1 / 100 
#
#     intermediars = []
#     err = []
#     n, d = A.shape
#
#     for k in range (1, T):
# stochastic
#         if comparison:
#             grad_real = gradient(A, W, p, m)
#             dist = grad_real - grad_aprox
#             err.append(np.linalg.norm(dist)) 
#
#         A -= (alpha / (k ** beta)) * grad_aprox
#
#         intermediars.append(f(A, W, p))
#
#     print(f"-> N = {N}, f* = {f(A, W, p)}")
#     return A, intermediars, err

def plot_sedii_depozite(coords, m, file_name, W = None, title = ''):
    n, d = coords.shape

    fig = plt.figure(figsize=(16, 9))

    if d == 2:
        ax = fig.add_subplot(111)
        if m > 0:
            ax.scatter(coords[:m, 0], coords[:m, 1], c='red', marker='*', label='sedii')
        if m < n:
            ax.scatter(coords[m:, 0], coords[m:, 1], c='blue', marker='o', label='depozite')

        if W is not None:
            for i in range(n):
                for j in range(i + 1, n):
                    if W[i][j]:
                        ax.plot([coords[i][0], coords[j][0]], [coords[i][1], coords[j][1]], 'k--', alpha=0.5)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title(title)
        ax.grid(True)
        ax.legend()

    elif d == 3:
        ax = fig.add_subplot(111, projection='3d')
        if m > 0:
            ax.scatter(coords[:m, 0], coords[:m, 1], coords[:m, 2], c='red', marker='*', label='sedii')
        if m < n:
            ax.scatter(coords[m:, 0], coords[m:, 1], coords[m:, 2], c='blue', marker='o', label='depozite')

        if W is not None:
            for i in range(n):
                for j in range(i + 1, n):
                    if W[i][j]:
                        ax.plot([coords[i][0], coords[j][0]], [coords[i][1], coords[j][1]], [coords[i][2], coords[j][2]], 'k--', alpha=0.5)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(title)
        ax.legend()

    plt.savefig(f'../plots/2_{file_name}.png', dpi = 300)
    plt.clf()

def plot_intermediars(intermediars_list, labels, title, file_name, legend_title, labeloy = 'f(x) intermediari'):
    plt.figure(figsize=(16, 9))
    for intermediars, label in zip(intermediars_list, labels):
        plt.semilogy(range(len(intermediars)), intermediars, label=label)
    plt.xlabel('Iteratii')
    plt.ylabel(labeloy)
    plt.title(title)
    plt.legend(title = legend_title)
    plt.grid()
    plt.yscale('log')
    plt.savefig(f'../plots/2_{file_name}.png', dpi = 300)
    plt.clf()

# a)
nr_sedii = 10
nr_depozite = 6
dimensiune = 2

p = 2

sedii = genereaza_sedii(nr_sedii, dimensiune)
depozite = init_depozite(nr_depozite, dimensiune)

# o fac fara cost ca sa fie mai usor de vizualizat rezultatul
# 2d
W = genereaza_matrice_adiacenta(nr_sedii, nr_depozite, ponderata = False)
A = np.vstack((sedii, depozite))

plot_sedii_depozite(A, nr_sedii, 'a_gd_init_2d', W, title = 'Conexiune sedii si coordonate initiale depozite')

A, intermediars, errors = gradient_descend(A, W, p, nr_sedii)

plot_intermediars([intermediars], ['f(x)'], f'Gradient Descend, p = {p}', f'a_gd_neponderat_p={p}', '')
plot_sedii_depozite(A, nr_sedii, f'a_result_gd_2d_p={p}', W, title = 'Conexiune sedii si coordonate finale depozite')

# 3d
# dimensiune = 3
#
# sedii = genereaza_sedii(nr_sedii, dimensiune)
# depozite = init_depozite(nr_depozite, dimensiune)
#
# W = genereaza_matrice_adiacenta(nr_sedii, nr_depozite, ponderata = False)
# A = np.vstack((sedii, depozite))
#
# plot_sedii_depozite(A, nr_sedii, 'a_gd_init_3d', W, title = 'Conexiune sedii si coordonate initiale depozite')
#
# A, intermediars, errors = gradient_descend(A, W, p, nr_sedii)
#
# plot_sedii_depozite(A, nr_sedii, 'a_result_gd_3d', W, title = 'Conexiune sedii si coordonate finale depozite')


# b
# nr_sedii = 10
# nr_depozite = 6
# dimensiune = 2
#
# p = 1
#
# sedii = genereaza_sedii(nr_sedii, dimensiune)
# depozite = init_depozite(nr_depozite, dimensiune)
#
# # o fac fara cost ca sa fie mai usor de vizualizat rezultatul
# # 2d
# W = genereaza_matrice_adiacenta(nr_sedii, nr_depozite, ponderata = False)
# A = np.vstack((sedii, depozite))
#
# plot_sedii_depozite(A, nr_sedii, 'a_gd_init_2d', W, title = 'Conexiune sedii si coordonate initiale depozite')
#
# A, intermediars, errors = gradient_descend(A, W, p, nr_sedii)
#
# plot_intermediars([intermediars], ['f(x)'], f'Gradient Descend, p = {p}', 'a_gd_neponderat', '')
# plot_sedii_depozite(A, nr_sedii, 'a_result_gd_2d', W, title = 'Conexiune sedii si coordonate finale depozite')
#

# c
# p = 1 / 2
#
# sedii = genereaza_sedii(nr_sedii, dimensiune)
# depozite = init_depozite(nr_depozite, dimensiune)
#
# W = genereaza_matrice_adiacenta(nr_sedii, nr_depozite, ponderata = False)
# A = np.vstack((sedii, depozite))
#
# A, intermediars, errors = gradient_descend(A, W, p, nr_sedii)
#
# plot_intermediars([intermediars], ['f(x)'], f'Gradient Descend, p = {p}', 'a_gd_neponderat', '')
#
# p = 3 / 2
#
# sedii = genereaza_sedii(nr_sedii, dimensiune)
# depozite = init_depozite(nr_depozite, dimensiune)
#
# W = genereaza_matrice_adiacenta(nr_sedii, nr_depozite, ponderata = False)
# A = np.vstack((sedii, depozite))
#
# A, intermediars, errors = gradient_descend(A, W, p, nr_sedii)
#
# plot_intermediars([intermediars], ['f(x)'], f'Gradient Descend, p = {p}', 'a_gd_neponderat', '')

