import numpy as np

def matmul_naive(A, B):
    """
    Наивное матричное умножение на Python, адаптированное под любые
    размерности и типы векторов/матриц из DPD-раннера.
    """
    # Гарантируем, что работаем с массивами NumPy
    A_np = np.asarray(A, dtype=np.complex128)
    B_np = np.asarray(B, dtype=np.complex128)
    
    # Извлекаем реальные размерности с учетом того, 1D это или 2D
    N = A_np.shape[0]
    M = A_np.shape[1] if A_np.ndim > 1 else 1
    
    # Определяем размерности для B
    if B_np.ndim == 1:
        M_b = B_np.shape[0]
        P = 1
    else:
        M_b, P = B_np.shape
        
    # Проверка на согласованность размерностей (для отладки, если что-то пойдет не так)
    if A_np.ndim > 1 and B_np.ndim > 1:
        assert M == M_b, f"Размерности не совпадают: A={A_np.shape}, B={B_np.shape}"
    elif A_np.ndim == 1 and B_np.ndim > 1:
        assert N == M_b, f"Размерности не совпадают: 1D A={A_np.shape}, 2D B={B_np.shape}"
        # Если А одномерный вектор-строка, то для корректного умножения N — это её длина (M)
        M = N
        N = 1

    # Выделяем буфер под результат нужной формы
    # Если на выходе должен быть вектор (P=1), делаем его плоским или матричным в зависимости от контекста
    C_np = np.zeros((N, P), dtype=np.complex128)

    # Меняем форму представлений для удобства тройного цикла "в лоб"
    A_2d = A_np.reshape(N, M)
    B_2d = B_np.reshape(M_b, P)

    # Честный наивный тройной цикл "в лоб" на чистом Python
    for i in range(N):
        for j in range(P):
            sum_val = complex(0.0, 0.0)
            for k in range(M):
                sum_val += complex(A_2d[i, k]) * complex(B_2d[k, j])
            C_np[i, j] = sum_val

    # Возвращаем ту форму, которую ожидает DPD раннер
    if B_np.ndim == 1 and C_np.shape[1] == 1:
        return C_np.flatten()
    if A_np.ndim == 1 and C_np.shape[0] == 1:
        return C_np.flatten()
        
    return C_np