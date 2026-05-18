import numpy as np

def matmul_perminov(A, B, out_C=None):
    """
    Инженерная реализация метода Перминова с минимизацией аллокаций.
    """
    n, k = A.shape
    m = B.shape[1]
    
    # 1. Подготовка выходного буфера
    if out_C is None:
        out_C = np.empty((n, m), dtype=A.dtype)
        
    # 2. Извлечение вещественных и мнимых частей без копирования (Views)
    A_re, A_im = A.real, A.imag
    B_re, B_im = B.real, B.imag

    # 3. Предвыделение временных буферов для промежуточных сумм
    # На RISC-V это будут временные регистры или стек
    tA = np.add(A_re, A_im)          # (ReA + ImA)
    tB = np.add(B_re, B_im)          # (ReB + ImB)

    # 4. Три вещественных матричных умножения (Ядро вычислений)
    M1 = np.matmul(A_re, B_re)       # ReA * ReB
    M2 = np.matmul(A_im, B_im)       # ImA * ImB
    M3 = np.matmul(tA, tB)           # (ReA + ImA) * (ReB + ImB)

    # 5. Сборка результата напрямую в компоненты out_C
    # Re(C) = M1 - M2
    np.subtract(M1, M2, out=out_C.real)
    
    # Im(C) = M3 - M1 - M2
    # Используем последовательные операции, чтобы не создавать новые массивы
    np.subtract(M3, M1, out=out_C.imag)
    np.subtract(out_C.imag, M2, out=out_C.imag)

    return out_C