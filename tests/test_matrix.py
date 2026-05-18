import numpy as np
import pytest
from src.matrix import multiply
from src.benchmarks.runner import create_gmp_matrix

def generate_gmp_test_data(n_samples=1200, p_order=7, m_depth=3, seed=42):
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 1, n_samples) + 1j * rng.normal(0, 1, n_samples)
    x /= np.std(x)

    # Генерация матрицы признаков напрямую через функцию раннера
    A = create_gmp_matrix(x, p_order=p_order, m_depth=m_depth)

    n_coeffs = A.shape[1]
    w_true = (rng.normal(0, 0.5, n_coeffs) + 1j * rng.normal(0, 0.5, n_coeffs))
    y_clean = A @ w_true
    noise_power = np.mean(np.abs(y_clean)**2) * 1e-5
    noise = np.sqrt(noise_power/2) * (rng.normal(0, 1, n_samples) + 1j * rng.normal(0, 1, n_samples))
    y = y_clean + noise
    return A, w_true, y

@pytest.mark.parametrize("method", ["naive", "winograd", "strassen", "perminov"])
@pytest.mark.parametrize("n_samples", [600, 1200])
def test_matmul_on_gmp_matrix(method, n_samples):
    A, w_true, y = generate_gmp_test_data(n_samples=n_samples)
    Ah = A.conj().T

    AhA_our = multiply(Ah, A, method=method)
    AhA_np = Ah @ A
    assert np.allclose(AhA_our, AhA_np, rtol=1e-10), f"{method}: A^H A mismatch"

    y_col = y.reshape(-1, 1)
    Ahy_our = multiply(Ah, y_col, method=method).flatten()
    Ahy_np = Ah @ y
    assert np.allclose(Ahy_our, Ahy_np, rtol=1e-10), f"{method}: A^H y mismatch"

    B = np.random.randn(A.shape[1], 16) + 1j * np.random.randn(A.shape[1], 16)
    C_our = multiply(A, B, method=method)
    C_np = A @ B
    assert np.allclose(C_our, C_np, rtol=1e-10), f"{method}: general matmul mismatch"

def test_ls_solution_via_matmul():
    A, w_true, y = generate_gmp_test_data(n_samples=1200)
    Ah = A.conj().T
    reg = 1e-4 * np.eye(A.shape[1])

    w_ref = np.linalg.solve(Ah @ A + reg, Ah @ y)

    methods = ["naive", "winograd", "strassen", "perminov"]
    for method in methods:
        AhA = multiply(Ah, A, method=method)
        Ahy = multiply(Ah, y.reshape(-1, 1), method=method).flatten()
        AhA_reg = AhA + reg
        w_test = np.linalg.solve(AhA_reg, Ahy)

        rel_err = np.linalg.norm(w_test - w_ref) / np.linalg.norm(w_ref)
        assert rel_err < 1e-8, f"LS with {method}: relative error {rel_err} too high"

        y_pred = A @ w_test
        y_ref_pred = A @ w_ref
        assert np.allclose(y_pred, y_ref_pred, rtol=1e-10), f"{method}: prediction mismatch"