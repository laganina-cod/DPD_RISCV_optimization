def matmul_naive(A, B):
    """
    Чистая реализация умножения матриц/векторов без numpy.
    Поддерживает: Matrix @ Matrix, Matrix @ Vector, Vector @ Matrix.
    """
    # 1. Определяем, является ли A вектором (1D)
    is_a_1d = not isinstance(A[0], (list, tuple))
    A_2d = [A] if is_a_1d else A
    
    n_a = len(A_2d)
    m_a = len(A_2d[0])

    # 2. Определяем, является ли B вектором (1D)
    is_b_1d = not isinstance(B[0], (list, tuple))
    if is_b_1d:
        # Превращаем [1, 2] в [[1], [2]] (столбец)
        B_2d = [[x] for x in B]
        m_b, p_b = len(B), 1
    else:
        B_2d = B
        m_b, p_b = len(B), len(B[0])

    if m_a != m_b:
        raise ValueError(f"Inconsistent dimensions: A_cols {m_a} != B_rows {m_b}")

    # 3. Само умножение
    result = [[0 for _ in range(p_b)] for _ in range(n_a)]
    for i in range(n_a):
        for j in range(p_b):
            for k in range(m_a):
                result[i][j] += A_2d[i][k] * B_2d[k][j]

    # 4. Возвращаем в ожидаемом формате
    if is_a_1d and is_b_1d:
        return result[0][0]  # Скаляр
    if is_a_1d:
        return result[0]     # Вектор-строка
    if is_b_1d:
        return [row[0] for row in result]  # Вектор-столбец (flatten)
    
    return result