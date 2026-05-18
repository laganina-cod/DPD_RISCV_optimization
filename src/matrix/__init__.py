from .naive import matmul_naive
from .winograd import matmul_winograd
from .strassen import matmul_strassen
from .perminov import matmul_perminov
import numpy as np

def multiply(a, b, method='perminov', leaf_size=32):
    """Единый интерфейс матричного умножения комплексных матриц."""
    a = np.asarray(a, dtype=np.complex128)
    b = np.asarray(b, dtype=np.complex128)

    methods = {
        'naive': matmul_naive,
        'winograd': matmul_winograd,
        'strassen': lambda x, y: matmul_strassen(x, y, leaf_size),
        'perminov': matmul_perminov
    }

    if method not in methods:
        raise ValueError(f"Метод {method} не поддерживается")

    return methods[method](a, b)