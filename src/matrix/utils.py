import numpy as np

def pad_to_power_of_two(a: np.ndarray, b: np.ndarray):
    """Дополняет матрицы нулями до ближайшей степени двойки (для Strassen)."""
    n1, k1 = a.shape
    k2, m1 = b.shape
    new_dim = 1 << (max(n1, k1, m1) - 1).bit_length()

    ap = np.zeros((new_dim, new_dim), dtype=np.complex128)
    bp = np.zeros((new_dim, new_dim), dtype=np.complex128)

    ap[:n1, :k1] = a
    bp[:k2, :m1] = b
    return ap, bp, n1, m1


def split_matrix(a: np.ndarray):
    """Разбивает квадратную матрицу на 4 квадранта."""
    mid = a.shape[0] // 2
    return (a[:mid, :mid], a[:mid, mid:],
            a[mid:, :mid], a[mid:, mid:])


def join_quadrants(c11, c12, c21, c22):
    """Собирает матрицу из 4 квадрантов."""
    return np.block([[c11, c12], [c21, c22]])


def trim_matrix(c: np.ndarray, rows: int, cols: int):
    """Удаляет padding, возвращая матрицу исходного размера."""
    return c[:rows, :cols].copy()