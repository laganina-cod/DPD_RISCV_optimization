import pytest
import numpy as np
from src.benchmarks.runner import create_gmp_matrix, align_signals
from src.dpd.solvers import (
    solver_naive, 
    solver_perminov, 
    solver_winograd,
    solver_cholesky_naive, 
    solver_cholesky_perminov, 
    solver_cholesky_winograd,
    qr_householder_solver
)

SOLVERS = [
    solver_naive, 
    solver_perminov, 
    solver_winograd,
    solver_cholesky_naive, 
    solver_cholesky_perminov, 
    solver_cholesky_winograd,
    qr_householder_solver
]

@pytest.mark.parametrize("solver", SOLVERS)
def test_solver_integration(solver):
    """Проверка базовой интеграции солверов без использования обертки класса."""
    np.random.seed(42)
    n_samples = 500
    x = np.random.randn(n_samples) + 1j * np.random.randn(n_samples)
    x /= np.max(np.abs(x)) 
    
    # Модель нелинейности
    y = x + 0.1 * np.abs(x)**2 * x + 0.05 * np.abs(x)**4 * x

    scale = np.max(np.abs(y))
    y_norm = y / scale
    x_norm = x / scale

    # Тестируем сквозную генерацию и расчет напрямую
    A_train = create_gmp_matrix(y_norm, p_order=3, m_depth=1)
    solver_result = solver(A_train, x_norm, reg=1e-4)
    w = solver_result[0] if isinstance(solver_result, tuple) else solver_result
    
    A_apply = create_gmp_matrix(x / scale, p_order=3, m_depth=1)
    x_pred = (A_apply @ w) * scale

    assert x_pred.shape == x.shape, f"Solver {solver.__name__} повредил размерность"
    assert not np.allclose(x, x_pred, rtol=1e-3), f"Solver {solver.__name__} выдал тривиальное тождественное решение"
    assert not np.any(np.isnan(x_pred)), f"Solver {solver.__name__} породил NaN"

@pytest.mark.parametrize("solver", SOLVERS)
def test_solver_convergence(solver):
    """Проверка сходимости МНК напрямую через функции раннера."""
    np.random.seed(42)
    x = np.random.randn(1000) + 1j * np.random.randn(1000)
    x /= np.max(np.abs(x))
    
    y_distorted = x + 0.2 * np.abs(x)**2 * x
    error_before = np.mean(np.abs(y_distorted - x)**2)
    
    scale = np.max(np.abs(y_distorted))
    A_train = create_gmp_matrix(y_distorted / scale, p_order=5, m_depth=2)
    
    solver_result = solver(A_train, x / scale, reg=1e-4)
    w = solver_result[0] if isinstance(solver_result, tuple) else solver_result
    
    A_apply = create_gmp_matrix(x / scale, p_order=5, m_depth=2)
    x_predistorted = (A_apply @ w) * scale
    
    y_corrected = predistorted_through_pa_model(x_predistorted)
    error_after = np.mean(np.abs(y_corrected - x)**2)
    
    assert error_after < error_before, f"Solver {solver.__name__} не уменьшил среднеквадратичную ошибку искажений"

def predistorted_through_pa_model(x_input):
    """Вспомогательная функция имитации полиномиального PA для тестов сходимости"""
    return x_input + 0.2 * np.abs(x_input)**2 * x_input

def test_winograd_odd_m():
    """Тест работы Винограда на нечетное число признаков (проверка остатка m % 2)."""
    x = np.random.randn(100) + 1j * np.random.randn(100)
    y = x * 1.1
    
    # p_order=2 дает степени [1], m_depth=3 -> 3 признака (нечетная матрица)
    A = create_gmp_matrix(y, p_order=2, m_depth=3)
    assert A.shape[1] % 2 != 0
    
    # Должно отработать без падения по Index Error во внутренних циклах Винограда
    res = solver_winograd(A, x, reg=1e-3)
    w = res[0] if isinstance(res, tuple) else res
    assert w is not None