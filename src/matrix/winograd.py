def matmul_winograd(A, B, out_C=None):
    # Вместо .shape используем встроенный len()
    # Работает и с объектами NumPy, и с обычными списками списков
    n = len(A)
    m = len(A[0])
    p = len(B[0])
    d = m // 2
    
    # Инициализация матрицы результата
    if out_C is None:
        out_C = [[0.0 for _ in range(p)] for _ in range(n)]
    else:
        # Если передан буфер (список списков), обнуляем его
        for i in range(n):
            for j in range(p):
                out_C[i][j] = 0.0

    # 1. Предварительный расчет факторов строк (row_factors)
    # row_f[i] = sum_{k=0}^{d-1} (A[i][2k] * A[i][2k+1])
    row_f = [0.0] * n
    for i in range(n):
        s = 0.0
        for k in range(d):
            s += A[i][2*k] * A[i][2*k+1]
        row_f[i] = s

    # 2. Предварительный расчет факторов столбцов (col_factors)
    # col_f[j] = sum_{k=0}^{d-1} (B[2k][j] * B[2k+1][j])
    col_f = [0.0] * p
    for j in range(p):
        s = 0.0
        for k in range(d):
            s += B[2*k][j] * B[2*k+1][j]
        col_f[j] = s

    # 3. Основной цикл вычислений
    for i in range(n):
        for j in range(p):
            # Базовое значение из суммы произведений по Винограду
            # sum_{k=0}^{d-1} (A[i][2k] + B[2k+1][j]) * (A[i][2k+1] + B[2k][j])
            val = 0.0
            for k in range(d):
                val += (A[i][2*k] + B[2*k+1][j]) * (A[i][2*k+1] + B[2*k][j])
            
            # Коррекция предвычисленными факторами
            out_C[i][j] = val - row_f[i] - col_f[j]

    # 4. Обработка нечетного размера (m % 2 != 0)
    if m % 2 == 1:
        last_col_idx = m - 1
        for i in range(n):
            term = A[i][last_col_idx]
            for j in range(p):
                out_C[i][j] += term * B[last_col_idx][j]

    return out_C