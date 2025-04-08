import numpy as np
import matplotlib.pyplot as plt

np.random.seed(22)

def gradient(A, x, b):
    return np.matmul(A.T, np.matmul(A, x) - b)

def f(A, x, b):
    return 1 / 2 * np.linalg.norm(np.matmul(A, x) - b) ** 2

def generate_A_b(m, n):
    r = min(m, n)
    sigma_max = 10 ** 5
    sigma_min = 1
    sigmas = np.linspace(sigma_max, sigma_min, r)

    Sigma = np.zeros((m, n))
    for i in range(r):
        Sigma[i, i] = sigmas[i]

    U, _ = np.linalg.qr(np.random.randn(m, m))  
    V, _ = np.linalg.qr(np.random.randn(n, n))

    A = U @ Sigma @ V.T

    b = np.random.randn(m)
    A_inv = np.linalg.pinv(A)
    b_proj = A @ A_inv @ b
    b_perp = b - b_proj
    sc = np.sqrt(2 * 1e3 + 1) / np.linalg.norm(b_perp)
    b = b_proj + sc * b_perp

    return A, b

def test(A, b):
    ATA = A.T @ A
    eigenvalues_ATA = np.linalg.eigvals(ATA)

    L = np.max(eigenvalues_ATA)
    sigma = np.min(eigenvalues_ATA)
    K = L / sigma
    print(f"Numarul de conditionare K = {K}")
    print(f"K > 10^6? {K > 10**6}")

    x_star = np.linalg.pinv(A) @ b
    V_star = 1 / 2 * np.linalg.norm(A @ x_star - b) ** 2
    print(f"V* = {V_star}")
    print(f"V* > 10^3? {V_star > 10**3}")

def gradient_descend(A, b, e = 1e3, x0 = None, alpha_method = "constant", alpha_constant = None, rho = 0.5, c = 0.5, max_iter = 100000):
    ATA = A.T @ A
    eigs_ATA = np.linalg.eigvals(ATA)

    L = np.max(eigs_ATA)

    if x0 is None:
        x = np.zeros(A.shape[1])
    else:
        x = x0

    if alpha_method == "constant" and alpha_constant is None:
        alpha_constant = 1 / L
        print(f"alfa constant: {alpha_constant}")

    intermediars = []
    alphas = []
    for i in range (max_iter):
        alpha_k = 0
        grad_x = gradient(A, x, b)
        if np.linalg.norm(grad_x) < e:
            print(f"minim gasit dupa {i} iteratii")
            break

        if alpha_method == "constant":
            alpha_k = alpha_constant

        # alpha in situatia asta poate fi calculat, derivata functiei dupa alpha da frumos
        elif alpha_method == "steepest":
            Ag = A @ grad_x
            residual = A @ x - b
            alpha_k = residual.T @ Ag / (np.linalg.norm(Ag) ** 2)

        elif alpha_method == "backtracking":
            alpha_k = 1.0
            t = 0
            while f(A, x - alpha_k * grad_x, b) > (f(A, x, b) - c * alpha_k * np.linalg.norm(grad_x) ** 2):
                alpha_k = rho * alpha_k
                if t > 100:
                    break
                t += 1

        x -= alpha_k * grad_x

        alphas.append(alpha_k)
        intermediars.append(f(A, x, b))

    print(f"f* = {f(A, x, b)}")
    return x, intermediars, alphas

def stochastic_gradient_descend(A, b, N, beta = 0.4, T = 3000, comparison = False):
    ATA = A.T @ A
    eigs_ATA = np.linalg.eigvals(ATA)

    L = np.max(eigs_ATA)

    # 0 < alpha < 1 / 2 * L. 
    # am incercat sa fac cu pas descrescator alpha / k, dar porneam de la un alpha prea mic (1 / L)
    alpha = 1 / L

    x = np.zeros(A.shape[1])
    intermediars = []
    err = []
    m = A.shape[0]

    for k in range (1, T):
        indices = np.random.choice(m, N, replace=False)
        A_selected = A[indices, :]
        b_selected = b[indices]

        grad_aprox = A_selected.T @ (A_selected @ x - b_selected) / N

        if comparison:
            grad_real = gradient(A, x, b)
            dist = grad_real - grad_aprox
            err.append(np.linalg.norm(dist)) 

        x -= (alpha / (k ** beta)) * grad_aprox

        intermediars.append(f(A, x, b))

    print(f"-> N = {N}, f* = {f(A, x, b)}")
    return x, intermediars, err

# ca sa calculez sk, voi calcula gradientul si voi adauga zgomot astfel incat eroarea gradientului aproximat sa se incadreze in limite
def stochastic_gradient_descend_d(A, b, T = 3000):
    test = True
    ATA = A.T @ A
    eigs_ATA = np.linalg.eigvals(ATA)

    L = np.max(eigs_ATA)

    alpha = 1 / (2 * L)

    x = np.zeros(A.shape[1])
    intermediars = []

    for _ in range (1, T):
        grad_real = gradient(A, x, b)

        noise = np.random.randn(A.shape[1]) 
        noise /= np.linalg.norm(noise)

        rho_k = np.random.uniform(0.01, 0.1)

        grad_aprox = grad_real + noise * rho_k

        dist = np.linalg.norm(grad_real - grad_aprox)
        if dist < 0.01 or dist > 0.1:
            test = False
            break

        x -= alpha * grad_aprox

        intermediars.append(f(A, x, b))

    print(f'-> test pentru distanta: {test}')
    print(f"-> f* = {f(A, x, b)}")
    return x, intermediars

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
    plt.savefig(f'../plots/1_{file_name}.png', dpi = 300)
    plt.clf()

def plot_alphas(alphas_list, labels, file_name):
    num_methods = len(alphas_list)
    ncols = 3 
    nrows = (num_methods + 1) // 3
    subplot_size = 5  
    
    _, axes = plt.subplots(nrows, ncols, figsize=(subplot_size * ncols, subplot_size * nrows))
    if num_methods == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for i, (alphas, label) in enumerate(zip(alphas_list, labels)):
        ax = axes[i]
        ax.plot(range(len(alphas)), alphas, label = label)
        ax.set_title(label)
        ax.set_xlabel('iteratii')
        ax.set_ylabel('α')
        ax.grid()
        ax.set_yscale('log')

    for i in range(num_methods, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig(f'../plots/1_{file_name}.png', dpi = 300)
    plt.clf()

# a)
m = 70
n = 50

A, b = generate_A_b(m, n)

print("-- Test A, b --")
test(A, b)


# b)
mat_A = np.random.rand(m, n)
vec_b = np.random.rand(m)

print("\n\n-- Gradient Descend --")
print("-> alpha_method constant")
x_constant, intermediars_constant, alphas_constant = gradient_descend(A, b, alpha_method="constant")
print("\n-> alpha_method steepest")
x_steepest, intermediars_steepest, alphas_steepest = gradient_descend(A, b, alpha_method="steepest")
print("\n-> alpha_method backtracking")
x_backtracking, intermediars_backtracking, alphas_backtracking = gradient_descend(A, b, alpha_method="backtracking")

intermediars_list = [intermediars_constant, intermediars_steepest, intermediars_backtracking]
alphas_list = [alphas_constant, alphas_steepest, alphas_backtracking]
labels = ['constant', 'steepest', 'backtracking']
title = 'Strategii de alegere a pasului de gradient pentru Gradient Descend'
file_name = 'gradient_descend_f_b'
legend_title = 'alpha_method'
plot_intermediars(intermediars_list, labels, title, file_name, legend_title)
plot_alphas(alphas_list, labels, 'gradient_descend_alpha_b')


# c)
print("\n\n-- Stochastic Gradient Descend --")
N = [1, 10, 30]
labels = []
intermediars_list = []
errors = []
for n in N:
    _, intermediars_stochastic, error = stochastic_gradient_descend(A, b, n, comparison = True)
    intermediars_list.append(intermediars_stochastic)
    errors.append(error)
    labels.append(f'N = {n}')
plot_intermediars(intermediars_list, labels, 'Stochastic Gradient Descend Number of Samples Comparison', 'stochastic_gradient_descend_c', 'num_samples')
plot_intermediars(errors, labels, 'Stochastic Gradient Descend Gradient Error Comparison', 'stochastic_gradient_descend_err_c', 'num_samples', labeloy = r'$\|\nabla f - g\|$')


# d)
print("\n\n-- Stochastic Gradient Descend (with gradient 0.01 < aproximation error < 0.1) --")
x_star, intermediars_stochastic_d = stochastic_gradient_descend_d(A, b)
plot_intermediars([intermediars_stochastic_d], ['gradient'], 'Stochastic Gradient Descend (with gradient 0.01 < aproximation error < 0.1)', 'stochastic_gradient_descend_d', '')
