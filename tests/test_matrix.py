import pytest
import random
from src.matrix.naive import matmul_naive
from src.matrix.strassen import matmul_strassen
from src.matrix.perminov import matmul_perminov
from src.matrix.winograd import matmul_winograd

METHODS = [
    ("naive", matmul_naive),
    ("strassen", matmul_strassen),
    ("perminov", matmul_perminov),
    ("winograd", matmul_winograd)
]

@pytest.mark.parametrize("name, func", METHODS)
class TestMatrixAlgorithms:
    def test_square_matrix(self, name, func):
        A = [[1.0, 2.0], [3.0, 4.0]]
        B = [[5.0, 6.0], [7.0, 8.0]]
        expected = [[19.0, 22.0], [43.0, 50.0]]
        assert func(A, B) == expected

# ВЫНОСИМ СЮДА: Эти тесты не должны быть внутри класса TestMatrixAlgorithms
def test_winograd_odd_dimensions():
    """Тест Винограда на нечетном размере m."""
    A = [[1., 2., 3.], [4., 5., 6.]]
    B = [[7., 8.], [9., 10.], [11., 12.]]
    expected = [[58., 64.], [139., 154.]]
    assert matmul_winograd(A, B) == expected

def test_strassen_padding():
    """Тест Штрассена на 3x3."""
    A = [[1.]*3 for _ in range(3)]
    assert len(matmul_strassen(A, A)) == 3