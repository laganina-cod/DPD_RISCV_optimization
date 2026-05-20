def matmul_perminov(A, B):
    # 1. Размеры матриц
    n, k, m = len(A), len(A[0]), len(B[0])

    # 2. Предварительный расчет сумм (tA и tB)
    tA = [[A[i][j].real + A[i][j].imag for j in range(k)] for i in range(n)]
    tB = [[B[i][j].real + B[i][j].imag for j in range(m)] for i in range(k)]

    # 3. Инициализация буферов для M1, M2, M3
    M1 = [[0.0] * m for _ in range(n)]
    M2 = [[0.0] * m for _ in range(n)]
    M3 = [[0.0] * m for _ in range(n)]

    # 4. Ядро вычислений: три умножения внутри одного тройного цикла
    # Используем порядок i -> l -> j для оптимального доступа к памяти
    for i in range(n):
        for l in range(k):
            # Кэшируем значения из A и tA во внутреннем цикле
            a_re = A[i][l].real
            a_im = A[i][l].imag
            a_sum = tA[i][l]
            
            for j in range(m):
                # Доступ к элементам B
                b_val = B[l][j]
                b_re = b_val.real
                b_im = b_val.imag

                # Накапливаем M1, M2 и M3 одновременно
                M1[i][j] += a_re * b_re
                M2[i][j] += a_im * b_im
                M3[i][j] += a_sum * tB[l][j]

    # 5. Сборка финальной комплексной матрицы
    # Re = M1 - M2, Im = M3 - M1 - M2
    return [
        [
            complex(M1[i][j] - M2[i][j], M3[i][j] - M1[i][j] - M2[i][j]) 
            for j in range(m)
        ] 
        for i in range(n)
    ]