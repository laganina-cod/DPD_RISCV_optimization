import numpy as np
from scipy.linalg import solve_triangular
from src.matrix.naive import matmul_naive  
from src.matrix.perminov import matmul_perminov
from src.matrix.strassen import matmul_strassen
from src.matrix.winograd import matmul_winograd

def solver_naive(A, y, reg=1e-3):
    Ah = A.conj().T
    matrix = matmul_naive(Ah, A) + reg * np.eye(A.shape[1])
    rhs = matmul_naive(Ah, y)
    w = np.linalg.solve(matrix, rhs)
    return w, None

def solver_perminov(A, y, reg=1e-3):
    Ah = A.conj().T
    matrix = matmul_perminov(Ah, A) + reg * np.eye(A.shape[1])
    rhs = matmul_perminov(Ah, y.reshape(-1, 1)).flatten()
    w = np.linalg.solve(matrix, rhs)
    return w, None

def solver_strassen(A, y, reg=1e-3):
    Ah = A.conj().T
    matrix = matmul_strassen(Ah, A) + reg * np.eye(A.shape[1])
    rhs = matmul_strassen(Ah, y.reshape(-1, 1)).flatten()
    w = np.linalg.solve(matrix, rhs)
    return w, None

def solver_winograd(A, y, reg=1e-3):
    Ah = A.conj().T
    matrix = matmul_winograd(Ah, A) + reg * np.eye(A.shape[1])
    rhs = matmul_winograd(Ah, y.reshape(-1, 1)).flatten()
    w = np.linalg.solve(matrix, rhs)
    return w, None

def solver_cholesky_naive(A, y, reg=1e-3):
    """Метод Холецкого со стандартным умножением матриц."""
    Ah = A.conj().T
    matrix = matmul_naive(Ah, A) + reg * np.eye(A.shape[1])
    rhs = matmul_naive(Ah, y)
    
    L = np.linalg.cholesky(matrix)
    z = solve_triangular(L, rhs, lower=True)
    w = solve_triangular(L.conj().T, z, lower=False)
    return w, None

def solver_cholesky_perminov(A, y, reg=1e-3):
    """Метод Холецкого с комплексным умножением по Перминову."""
    Ah = A.conj().T
    matrix = matmul_perminov(Ah, A) + reg * np.eye(A.shape[1])
    rhs = matmul_perminov(Ah, y.reshape(-1, 1)).flatten()
    
    L = np.linalg.cholesky(matrix)
    z = solve_triangular(L, rhs, lower=True)
    w = solve_triangular(L.conj().T, z, lower=False)
    return w, None

def solver_cholesky_strassen(A, y, reg=1e-3):
    """Метод Холецкого с блочным умножением Штрассена."""
    Ah = A.conj().T
    matrix = matmul_strassen(Ah, A) + reg * np.eye(A.shape[1])
    rhs = matmul_strassen(Ah, y.reshape(-1, 1)).flatten()
    
    L = np.linalg.cholesky(matrix)
    z = solve_triangular(L, rhs, lower=True)
    w = solve_triangular(L.conj().T, z, lower=False)
    return w, None

def solver_cholesky_winograd(A, y, reg=1e-3):
    """Метод Холецкого с алгоритмом Винограда."""
    Ah = A.conj().T
    matrix = matmul_winograd(Ah, A) + reg * np.eye(A.shape[1])
    rhs = matmul_winograd(Ah, y.reshape(-1, 1)).flatten()
    
    L = np.linalg.cholesky(matrix)
    z = solve_triangular(L, rhs, lower=True)
    w = solve_triangular(L.conj().T, z, lower=False)
    return w, None

def qr_householder_solver(A, y, reg=1e-3):
    """Решение через QR-разложение (наиболее устойчивое к плохо обусловленным матрицам)."""
    # Добавляем регуляризацию прямо в матрицу A (расширенная система)
    reg_matrix = np.sqrt(reg) * np.eye(A.shape[1])
    A_ext = np.vstack([A, reg_matrix])
    y_ext = np.concatenate([y, np.zeros(A.shape[1])])
    
    Q, R = np.linalg.qr(A_ext)
    w = solve_triangular(R, Q.conj().T @ y_ext, lower=False)
    return w, None

