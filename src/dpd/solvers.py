import math

# --- Импорты ваших алгоритмов ---
from src.matrix.naive import matmul_naive 
from src.matrix.perminov import matmul_perminov
from src.matrix.strassen import matmul_strassen
from src.matrix.winograd import matmul_winograd 

# --- Универсальные утилиты ---

def to_matrix(obj):
    if hasattr(obj, 'tolist'): obj = obj.tolist()
    if not isinstance(obj, list): return [[obj]]
    if not isinstance(obj[0], list): return [obj]
    return obj

def to_vector(obj):
    if hasattr(obj, 'tolist'): obj = obj.tolist()
    if isinstance(obj, list) and len(obj) > 0 and isinstance(obj[0], list):
        return [row[0] for row in obj]
    return obj

def identity_matrix(n, scale=1.0):
    return [[(scale if i == j else 0.0) for j in range(n)] for i in range(n)]

def add_matrices(A, B):
    return [[A[i][j] + B[i][j] for j in range(len(A[0]))] for i in range(len(A))]

def transpose_conj(A):
    return [[(A[j][i].conjugate() if hasattr(A[j][i], 'conjugate') else A[j][i]) 
             for j in range(len(A))] for i in range(len(A[0]))]

# --- Базовые математические алгоритмы ---

def cholesky_pure(A):
    n = len(A)
    L = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1):
            s = sum(L[i][k] * (L[j][k].conjugate() if hasattr(L[j][k], 'conjugate') else L[j][k]) 
                    for k in range(j))
            if i == j:
                val = A[i][i] - s
                L[i][j] = math.sqrt(max(val.real if hasattr(val, 'real') else val, 0.0))
            else:
                L[i][j] = (1.0 / L[j][j] * (A[i][j] - s))
    return L

def solve_triangular_pure(L, b, lower=True):
    n = len(L)
    x = [0.0] * n
    if lower:
        for i in range(n):
            s = sum(L[i][j] * x[j] for j in range(i))
            x[i] = (b[i] - s) / L[i][i]
    else:
        for i in range(n - 1, -1, -1):
            s = sum((L[j][i].conjugate() if hasattr(L[j][i], 'conjugate') else L[j][i]) * x[j] 
                    for j in range(i + 1, n))
            divisor = (L[i][i].conjugate() if hasattr(L[i][i], 'conjugate') else L[i][i])
            x[i] = (b[i] - s) / divisor
    return x

def gauss_solve(A, b):
    n = len(A)
    M = [A[i] + [b[i]] for i in range(n)]
    for i in range(n):
        max_el, max_row = abs(M[i][i]), i
        for k in range(i + 1, n):
            if abs(M[k][i]) > max_el:
                max_el, max_row = abs(M[k][i]), k
        M[i], M[max_row] = M[max_row], M[i]
        for k in range(i + 1, n):
            c = -M[k][i] / M[i][i]
            for j in range(i, n + 1):
                M[k][j] = 0 if i == j else M[k][j] + c * M[i][j]
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        x[i] = M[i][n] / M[i][i]
        for k in range(i - 1, -1, -1):
            M[k][n] -= M[k][i] * x[i]
    return x

# --- Решатели Гаусса ---

def solver_naive(A, y, reg=1e-3):
    A_m, y_v = to_matrix(A), to_vector(y)
    Ah = transpose_conj(A_m)
    matrix = add_matrices(matmul_naive(Ah, A_m), identity_matrix(len(Ah), reg))
    rhs = to_vector(matmul_naive(Ah, [[v] for v in y_v]))
    return gauss_solve(matrix, rhs)

def solver_perminov(A, y, reg=1e-3):
    A_m, y_v = to_matrix(A), to_vector(y)
    Ah = transpose_conj(A_m)
    matrix = add_matrices(matmul_perminov(Ah, A_m), identity_matrix(len(Ah), reg))
    rhs = to_vector(matmul_perminov(Ah, [[v] for v in y_v]))
    return gauss_solve(matrix, rhs)

def solver_strassen(A, y, reg=1e-3):
    A_m, y_v = to_matrix(A), to_vector(y)
    Ah = transpose_conj(A_m)
    matrix = add_matrices(matmul_strassen(Ah, A_m), identity_matrix(len(Ah), reg))
    rhs = to_vector(matmul_strassen(Ah, [[v] for v in y_v]))
    return gauss_solve(matrix, rhs)

def solver_winograd(A, y, reg=1e-3):
    A_m, y_v = to_matrix(A), to_vector(y)
    Ah = transpose_conj(A_m)
    matrix = add_matrices(matmul_winograd(Ah, A_m), identity_matrix(len(Ah), reg))
    rhs = to_vector(matmul_winograd(Ah, [[v] for v in y_v]))
    return gauss_solve(matrix, rhs)

# --- Решатели Холецкого ---

def solver_cholesky_perminov(A, y, reg=1e-3):
    A_m, y_v = to_matrix(A), to_vector(y)
    Ah = transpose_conj(A_m)
    matrix = add_matrices(matmul_perminov(Ah, A_m), identity_matrix(len(Ah), reg))
    rhs = to_vector(matmul_perminov(Ah, [[v] for v in y_v]))
    L = cholesky_pure(matrix)
    y_temp = solve_triangular_pure(L, rhs, lower=True)
    return solve_triangular_pure(L, y_temp, lower=False)

def solver_cholesky_naive(A, y, reg=1e-3):
    """Решатель Холецкого с использованием наивного умножения."""
    A_m, y_v = to_matrix(A), to_vector(y)
    Ah = transpose_conj(A_m)
    # Формируем матрицу нормальных уравнений через наивное умножение
    matrix = add_matrices(matmul_naive(Ah, A_m), identity_matrix(len(Ah), reg))
    rhs = to_vector(matmul_naive(Ah, [[v] for v in y_v]))
    
    L = cholesky_pure(matrix)
    # Прямой и обратный ход
    y_temp = solve_triangular_pure(L, rhs, lower=True)
    return solve_triangular_pure(L, y_temp, lower=False)

def solver_cholesky_winograd(A, y, reg=1e-3):
    """Решатель Холецкого с использованием алгоритма Винограда."""
    A_m, y_v = to_matrix(A), to_vector(y)
    Ah = transpose_conj(A_m)
    # Формируем матрицу нормальных уравнений через Винограда
    matrix = add_matrices(matmul_winograd(Ah, A_m), identity_matrix(len(Ah), reg))
    rhs = to_vector(matmul_winograd(Ah, [[v] for v in y_v]))
    
    L = cholesky_pure(matrix)
    y_temp = solve_triangular_pure(L, rhs, lower=True)
    return solve_triangular_pure(L, y_temp, lower=False)

def solver_cholesky_strassen(A, y, reg=1e-3):
    """Решатель Холецкого с использованием алгоритма Штрассена."""
    A_m, y_v = to_matrix(A), to_vector(y)
    Ah = transpose_conj(A_m)
    # Формируем матрицу нормальных уравнений через Штрассена
    matrix = add_matrices(matmul_strassen(Ah, A_m), identity_matrix(len(Ah), reg))
    rhs = to_vector(matmul_strassen(Ah, [[v] for v in y_v]))
    
    L = cholesky_pure(matrix)
    y_temp = solve_triangular_pure(L, rhs, lower=True)
    return solve_triangular_pure(L, y_temp, lower=False)

# --- Решатель QR Householder ---

def qr_householder_solver(A, y, reg=1e-3):
    A_l, y_l = to_matrix(A), to_vector(y)
    n_cols = len(A_l[0])
    R = [row[:] for row in A_l]
    y_ext = y_l[:] + [0.0] * n_cols
    lmbda = math.sqrt(reg)
    for i in range(n_cols):
        R.append([lmbda if j == i else 0.0 for j in range(n_cols)])
    
    curr_rows = len(R)
    for i in range(n_cols):
        col_norm = math.sqrt(sum(abs(R[j][i])**2 for j in range(i, curr_rows)))
        if col_norm < 1e-18: continue
        alpha = R[i][i]
        phase = alpha / abs(alpha) if abs(alpha) > 1e-18 else 1.0
        sigma = phase * col_norm
        v = [(R[j][i] if j > i else alpha + sigma) for j in range(i, curr_rows)]
        v_norm_sq = sum(abs(vk)**2 for vk in v)
        if v_norm_sq < 1e-18: continue
        for k in range(i, n_cols):
            dot_vr = sum(v[j-i].conjugate() * R[j][k] for j in range(i, curr_rows))
            gamma = (2.0 * dot_vr) / v_norm_sq
            for j in range(i, curr_rows): R[j][k] -= gamma * v[j-i]
        dot_vy = sum(v[j-i].conjugate() * y_ext[j] for j in range(i, curr_rows))
        gamma_y = (2.0 * dot_vy) / v_norm_sq
        for j in range(i, curr_rows): y_ext[j] -= gamma_y * v[j-i]
            
    w = [0.0] * n_cols
    for i in range(n_cols - 1, -1, -1):
        if abs(R[i][i]) > 1e-18:
            w[i] = (y_ext[i] - sum(R[i][j] * w[j] for j in range(i + 1, n_cols))) / R[i][i]
    return w