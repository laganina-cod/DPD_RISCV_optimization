import numpy as np

def matmul_winograd(A, B, out_C=None):
    n, m = A.shape
    p = B.shape[1]
    d = m // 2
    
    if out_C is None:
        out_C = np.zeros((n, p), dtype=A.dtype)
    else:
        out_C.fill(0) # Очищаем буфер

    # Инженерная оптимизация факторов: 
    # Считаем их один раз и используем как read-only массивы
    # row_factor = sum(A[i, 2k] * A[i, 2k+1])
    # col_factor = sum(B[2k, j] * B[2k+1, j])
    
    # Векторизованный расчет факторов без лишних копий
    row_f = np.sum(A[:, 0:2*d:2] * A[:, 1:2*d:2], axis=1)
    col_f = np.sum(B[0:2*d:2, :] * B[1:2*d:2, :], axis=0)

    # В основном цикле мы имитируем работу блоками
    # На RISC-V здесь будет итерация по L1-кэшу
    for i in range(n):
        # Векторизованная сумма по k
        # На языке С это будет: C[i,j] += (A[i, 2k] + B[2k+1, j]) * (A[i, 2k+1] + B[2k, j])
        A_even = A[i, 0:2*d:2]
        A_odd  = A[i, 1:2*d:2]
        B_even = B[0:2*d:2, :]
        B_odd  = B[1:2*d:2, :]
        
        # Инженерный трюк: вычисляем строку сразу для всех столбцов j
        # Это минимизирует промахи по кэшу для строки A[i]
        out_C[i, :] = np.sum((A_even[:, None] + B_odd) * (A_odd[:, None] + B_even), axis=0)
        
        # Финальная коррекция факторами
        out_C[i, :] -= (row_f[i] + col_f)

    if m % 2 == 1:
        # Внешнее произведение (rank-1 update)
        out_C += np.outer(A[:, -1], B[-1, :])

    return out_C