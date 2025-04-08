import numpy as np
import matplotlib.pyplot as plt

np.random.seed(22)

def f(coords, W, p):
    n = coords.shape[0]
    
    sigma = 0.0
    
    for i in range(n):
        for j in range(i + 1, n):
            if W[i][j]:  
                dist = np.linalg.norm(coords[i] - coords[j])
                sigma += W[i][j] * dist ** p
    
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
def genereaza_matrice_adiacenta(m, n, connection_prob=0.7):
    total_points = m + n
    
    W = np.zeros((total_points, total_points), dtype=int)
    
    for i in range(m):  
        for j in range(m, total_points):  
            if np.random.rand() < connection_prob:
                W[i][j] = 1
                W[j][i] = 1  
    
    return W


def plot_sedii_depozite(coords, m, W = None, title = ''):
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
    
    plt.show()

nr_sedii = 10
nr_depozite = 4
dimensiune = 2

sedii = genereaza_sedii(nr_sedii, dimensiune)
depozite = init_depozite(nr_depozite, dimensiune)

W = genereaza_matrice_adiacenta(nr_sedii, nr_depozite)
A = np.vstack((sedii, depozite))

# plot_sedii_depozite(sedii, nr_sedii, W, title = 'Sedii')
# plot_sedii_depozite(A, nr_sedii, W, title = 'Conexiune sedii si coordonate initiale depozite')
print(f(A, W, 2))
print(gradient(A, W, 2, nr_sedii))
