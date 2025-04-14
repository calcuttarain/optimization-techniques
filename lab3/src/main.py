import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp
import scipy

np.random.seed(22)

def generate_time_series(n, sigma, p):
    noise = np.random.normal(0, sigma ** 2, n)
    p_tresholds = np.random.rand(n)
    v = np.random.uniform(low = -1, high = 1, size = n)

    x = np.zeros(n)
    x[0] = 0

    for i in range (1, n):
        if p_tresholds[i] < p:
            v[i] = v[i - 1]

        x[i] = x[i - 1] + v[i]

    y = x + noise 

    return x, y, noise, v

def generate_difference_matrix(n):
    diagonals = [np.ones(n - 2), -2 * np.ones(n - 2), np.ones(n - 2)]
    offsets = [0, 1, 2]  
    
    D = scipy.sparse.diags(diagonals, offsets, shape=(n - 2, n)).toarray()
    return D

def filter_function(x, y, D, rho):
    fidelity = 1 / 2 * (cp.norm(x - y, 2) ** 2)
    smoothness = rho * cp.norm(D @ x, 1)

    filter = fidelity + smoothness

    return filter

'''
ca sa mearga mgp pentru n > 10 ** 4, matricea D este prea mare pentru a fi stocata in memorie 
trebuie sa aplic functia de convolutie pe x echivalenta cu Dx
similar si pentru D.Tx 
'''
def compute_DTx(x):
    n = len(x) + 2
    DTx = np.zeros(n)
    for i in range(len(x)):
        DTx[i] += x[i]        
        DTx[i+1] += -2 * x[i]
        DTx[i+2] += x[i]    
    return DTx

def compute_Dx(x):
    n = len(x)
    Dx = np.zeros(n - 2)  
    for i in range(n - 2):
        Dx[i] = x[i] - 2 * x[i + 1] + x[i + 2]  
    return Dx

def gradient(x, y):
    return compute_Dx(compute_DTx(x) - y)

def projection(x, rho):
    return np.maximum(-rho, np.minimum(rho, x))

def g(x, alpha, grad_x, rho):
    return 1 / alpha * (x - projection(x - alpha * grad_x, rho))

def f_lagrange(miu, y):
    return (- 1 / 2) * np.linalg.norm(compute_DTx(miu)) ** 2 + miu.T @ compute_Dx(y)

def mgp(y, rho, n, e = 1e-2, p = 0.5, c = 0.5, max_iter = 1000):
    miu = np.zeros(n - 2)
    miu = projection(miu, rho)

    for i in range (max_iter):
        grad_miu = gradient(miu, y)

        alpha_k = 1.0
        t = 0

        f_x_k = f_lagrange(miu, y)
        f_x_k_next = f_lagrange(projection(miu - alpha_k * grad_miu, rho), y)
        g_x_k = g(miu, alpha_k, grad_miu, rho)

        while f_x_k_next < (f_x_k - c * alpha_k * np.linalg.norm(g_x_k) ** 2):
            alpha_k = p * alpha_k

            miu_proj = projection(miu - alpha_k * grad_miu, rho)
            f_x_k_next = f_lagrange(miu_proj, y)
            g_x_k = 1 / alpha_k * (miu - miu_proj)

            if t > 100:
                break
            t += 1

        miu = projection(miu - alpha_k * grad_miu, rho)

        if np.linalg.norm(g_x_k) < e:
            print(f"minim gasit dupa {i} iteratii")
            break

    return miu

def solve_x_hp(y, rho, n, D):
    A = np.eye(n) + 2 * rho * D.T @ D
    '''
    A este o matrice pentadiagonala simetrica pozitiv definita si inversabila.
    Se inmulteste ecuatia la stanga cu A si se rezolva sistemul Ax = y cu factorizarea A = LU in O(n).
    '''
    L = np.zeros((n, n))
    U = np.zeros((n, n))
    
    for i in range(n):
        if i == 0:
            U[0, 0] = A[0, 0]
            U[0, 1] = A[0, 1]
            if n > 2:
                U[0, 2] = A[0, 2]
        elif i == 1:
            L[1, 0] = A[1, 0] / U[0, 0]
            U[1, 1] = A[1, 1] - L[1, 0] * U[0, 1]
            U[1, 2] = A[1, 2] - L[1, 0] * U[0, 2] if n > 2 else 0
            if n > 3:
                U[1, 3] = A[1, 3]
        else:
            if i >= 2:
                L[i, i-2] = A[i, i-2] / U[i-2, i-2]
            L[i, i-1] = (A[i, i-1] - (L[i, i-2] * U[i-2, i-1] if i >= 2 else 0)) / U[i-1, i-1]
            U[i, i] = A[i, i] - L[i, i-1] * U[i-1, i] - (L[i, i-2] * U[i-2, i] if i >= 2 else 0)
            if i < n-1:
                U[i, i+1] = A[i, i+1] - L[i, i-1] * U[i-1, i+1] - (L[i, i-2] * U[i-2, i+1] if i >= 2 else 0)
            if i < n-2:
                U[i, i+2] = A[i, i+2] - L[i, i-1] * U[i-1, i+2] - (L[i, i-2] * U[i-2, i+2] if i >= 2 else 0)
    
    z = np.zeros(n)
    for i in range(n):
        if i == 0:
            z[0] = y[0]
        elif i == 1:
            z[1] = y[1] - L[1, 0] * z[0]
        else:
            z[i] = y[i] - L[i, i-1] * z[i-1] - (L[i, i-2] * z[i-2] if i >= 2 else 0)
    
    x = np.zeros(n)
    for i in range(n-1, -1, -1):
        if i == n-1:
            x[i] = z[i] / U[i, i]
        elif i == n-2:
            x[i] = (z[i] - U[i, i+1] * x[i+1]) / U[i, i]
        else:
            x[i] = (z[i] - U[i, i+1] * x[i+1] - U[i, i+2] * x[i+2]) / U[i, i]
    
    return x

def plot_trend_time_series(trend, time_series, n, labels, title, file_name):
    t = np.arange(n)
    
    plt.figure(figsize=(16, 9))
    
    plt.plot(t, time_series, label = labels[1], color='red')  
    plt.plot(t, trend, label = labels[0], color='black')
    
    plt.title(title)  
    plt.xlabel('time')
    plt.ylabel('value')
    plt.grid(True)
    plt.legend()
    
    plt.savefig(f'../plots/{file_name}.png', dpi=300)
    
    plt.close()

def plot_x_hp_vs_x_l1(x_hp, x_l1, time_series, n):
    t = np.arange(n)
    
    plt.figure(figsize=(16, 9))
    
    plt.plot(t, time_series, label = r'$y_t$', color='red')  
    plt.plot(t, x_l1, label = r'$\hat{x}_{\ell_1}$', color='black')
    plt.plot(t, x_hp, label = '$x_{HP}$', color='purple')
    
    plt.title('x_HP vs x_l1')  
    plt.xlabel('time')
    plt.ylabel('value')
    plt.grid(True)
    plt.legend()
    
    plt.savefig('../plots/e_x_hp_vs_x_l1_2.png', dpi=300)
    
    plt.close()

def plot_cvxpy_results(results, labels, n):
    t = np.arange(n)
    
    plt.figure(figsize=(16, 9))
    
    for result, label in zip(results, labels):
        plt.plot(t, result, label = label)  
    
    plt.title('CVXPY Resutls')  
    plt.xlabel('time')
    plt.ylabel('value')
    plt.grid(True)
    plt.legend()
    
    plt.savefig('../plots/b_cvxpy_results.png', dpi=300)
    
    plt.close()


# a
n = 10000
sigma = 2.5 
probability = 0.9

trend, time_series, noise, v = generate_time_series(n, sigma, probability)

plot_trend_time_series(trend, time_series, n, [r'$x_t$', r'$y_t$'], 'Trend vs Time Series', 'a_trend_noisy')


# b
D = generate_difference_matrix(n)

rho_list = [1, 50]
results = []
labels = []

for rho in rho_list:
    x = cp.Variable(n)

    obj = cp.Minimize(filter_function(x, time_series, D, rho))
    prob = cp.Problem(obj)
    prob.solve()

    print(f'\n-> rho = {rho}')
    print("status:", prob.status)
    print("optimal value:", prob.value)

    results.append(x.value)
    labels.append(r'$\rho = ' + f'{rho}$')

plot_cvxpy_results(results, labels, n)


# c
rho = 10
miu_star = mgp(time_series, rho, n)
x_l1 = time_series - compute_DTx(miu_star)

plot_trend_time_series(x_l1, time_series, n, [r'$\hat{x}_{\ell_1}$', r'$y_t$'], f'Metoda Gradient Proiectat pentru rho = {rho}', 'c_mgp')


# d
n = 1000
sigma = 2.5 
probability = 0.9

trend, time_series, noise, v = generate_time_series(n, sigma, probability)
D = generate_difference_matrix(n)

rho = 10

x_hp = solve_x_hp(time_series, rho, n, D)
plot_trend_time_series(x_hp, time_series, n, [r'$x_{HP}$', r'$y_t$'], f'Solutia Modelului Hodrick-Prescott pentru rho = {rho}', 'd_x_hp')


# e
miu_star = mgp(time_series, rho, n)
x_l1 = time_series - compute_DTx(miu_star)

plot_trend_time_series(x_hp, x_l1, n, ['$x_{HP}$', r'$\hat{x}_{\ell_1}$'], f'Solutia Modelului HP vs Solutia l1 pentru rho = {rho}', 'e_x_hp_vs_x_l1_1')
plot_x_hp_vs_x_l1(x_hp, x_l1, time_series, n)

'''
x_hp e mai smooth, penalizeaza mai mult din diferentele mai mari spre deosebire de x_l1
provine din comportamentul diferit pe care il are in general norma 1 fata de norma 2
'''
