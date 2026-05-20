import math
import pytest
import numpy as np
from src.benchmarks.runner import create_gmp_matrix
from src.dpd.solvers import (
    solver_naive, solver_perminov, solver_strassen, solver_winograd,
    solver_cholesky_perminov, qr_householder_solver
)

# Список актуальных солверов из вашей реализации
SOLVERS = [
    solver_naive, 
    solver_perminov, 
    solver_strassen, 
    solver_winograd,
    solver_cholesky_perminov, 
    qr_householder_solver
]

def to_list_pure(data):
    if hasattr(data, 'tolist'): return data.tolist()
    return data

def apply_weights_pure(A, w):
    """Применение весов: x_pred = A @ w."""
    res = []
    for row in A:
        val = sum(row[i] * w[i] for i in range(len(w)))
        res.append(val)
    return res

def calculate_mse_pure(y_true, y_pred):
    n = len(y_true)
    total_error = sum(abs(y_true[i] - y_pred[i])**2 for i in range(n))
    return total_error / n

@pytest.mark.parametrize("solver", SOLVERS)
def test_solver_integration(solver):
    """Проверка, что солвер возвращает корректный вектор весов."""
    n_samples = 100
    x = [(math.sin(i) + 1j*math.cos(i)) * 0.5 for i in range(n_samples)]
    y = [val + 0.1 * val * (abs(val)**2) for val in x]

    # Матрица GMP (теперь работаем напрямую с результатом)
    A_train = to_list_pure(create_gmp_matrix(y, p_order=3, m_depth=1))
    
    # ВАЖНО: теперь просто получаем w, без проверок на кортежи
    w = solver(A_train, x, reg=1e-4)

    x_pred = apply_weights_pure(A_train, w)

    assert len(x_pred) == len(x)
    assert not any(math.isnan(v.real) or math.isnan(v.imag) for v in x_pred)

@pytest.mark.parametrize("solver", SOLVERS)
def test_solver_convergence(solver):
    """Проверка сходимости: ошибка после DPD должна быть меньше, чем до."""
    np.random.seed(42)
    n = 400
    x = ((np.random.randn(n) + 1j*np.random.randn(n)) * 0.5).tolist()
    
    def pa_model(signal):
        return [v + 0.2 * v * (abs(v)**2) for v in signal]
    
    y_distorted = pa_model(x)
    error_before = calculate_mse_pure(x, y_distorted)
    
    A_train = to_list_pure(create_gmp_matrix(np.array(y_distorted), p_order=3, m_depth=1))
    
    # Получаем веса напрямую
    w = solver(A_train, x, reg=1e-5) 
    
    x_recovered = apply_weights_pure(A_train, to_list_pure(w))
    error_after = calculate_mse_pure(x, x_recovered)
    
    # Для всех методов ожидаем значительное снижение ошибки
    assert error_after < error_before * 0.1, f"Solver {solver.__name__} не обеспечил сходимость"

def test_winograd_odd_m_manual():
    """Тест специфики Винограда на нечетном количестве столбцов."""
    x = [1.1, 0.55, 0.22, 0.11, 0.055]
    A = [[1.0, 0.1, 0.01], 
         [0.5, 0.05, 0.005], 
         [0.2, 0.02, 0.002], 
         [0.1, 0.01, 0.001],
         [0.05, 0.005, 0.0005]]
    
    w = solver_winograd(A, x, reg=1e-3)
    assert len(w) == 3