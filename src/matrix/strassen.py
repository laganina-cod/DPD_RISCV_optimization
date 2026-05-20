# --- Вспомогательные функции (Utility) ---


def add_pure(A, B):
    """Поэлементное сложение матриц."""
    return [[A[i][j] + B[i][j] for j in range(len(A[0]))] for i in range(len(A))]


def sub_pure(A, B):
    """Поэлементное вычитание матриц."""
    return [[A[i][j] - B[i][j] for j in range(len(A[0]))] for i in range(len(A))]


def matmul_base_case(A, B):
    """
    Наивное умножение O(n^3). 
    Используется как база рекурсии для Штрассена.
    """
    n, m, p = len(A), len(A[0]), len(B[0])
    C = [[0.0 for _ in range(p)] for _ in range(n)]
    for i in range(n):
        for k in range(m):
            temp = A[i][k]
            for j in range(p):
                C[i][j] += temp * B[k][j]
    return C


def pad_matrix(M, new_size):
    """Дополнение матрицы нулями до квадрата со стороной new_size."""
    new_M = [[0.0 for _ in range(new_size)] for _ in range(new_size)]
    for i in range(len(M)):
        for j in range(len(M[0])):
            new_M[i][j] = M[i][j]
    return new_M


# --- Основная логика ---


def matmul_strassen(A, B):
    """
    Интерфейсная функция: подготавливает размеры и вызывает рекурсию.
    """
    n = len(A)
    m = len(A[0])
    p = len(B[0])

    # Если матрица слишком мала или не квадратная, решаем, нужно ли дополнение
    max_dim = max(n, m, p)
    
    # База рекурсии для очень маленьких матриц
    if max_dim <= 4:
        return matmul_base_case(A, B)

    # Вычисляем ближайшую степень двойки
    pow2 = 1
    while pow2 < max_dim:
        pow2 *= 2

    # Выполняем Padding, если размеры не соответствуют 2^n
    if n != pow2 or m != pow2 or p != pow2:
        A_padded = pad_matrix(A, pow2)
        B_padded = pad_matrix(B, pow2)
        C_padded = _strassen_recursive(A_padded, B_padded)
        # Обрезаем результат до оригинального размера (n x p)
        return [row[:p] for row in C_padded[:n]]
    else:
        return _strassen_recursive(A, B)


def _strassen_recursive(A, B):
    """
    Рекурсивное ядро алгоритма Штрассена.
    """
    n = len(A)
    
    # Порог перехода на наивное умножение. 
    # На RISC-V подбирается экспериментально (обычно 16-64).
    if n <= 16:
        return matmul_base_case(A, B)

    mid = n // 2

    # Разбиение на четверти
    # A
    a11 = [row[:mid] for row in A[:mid]]
    a12 = [row[mid:] for row in A[:mid]]
    a21 = [row[:mid] for row in A[mid:]]
    a22 = [row[mid:] for row in A[mid:]]
    # B
    b11 = [row[:mid] for row in B[:mid]]
    b12 = [row[mid:] for row in B[:mid]]
    b21 = [row[:mid] for row in B[mid:]]
    b22 = [row[mid:] for row in B[mid:]]

    # 7 формул Штрассена
    m1 = _strassen_recursive(add_pure(a11, a22), add_pure(b11, b22))
    m2 = _strassen_recursive(add_pure(a21, a22), b11)
    m3 = _strassen_recursive(a11, sub_pure(b12, b22))
    m4 = _strassen_recursive(a22, sub_pure(b21, b11))
    m5 = _strassen_recursive(add_pure(a11, a12), b22)
    m6 = _strassen_recursive(sub_pure(a21, a11), add_pure(b11, b12))
    m7 = _strassen_recursive(sub_pure(a12, a22), add_pure(b21, b22))

    # Компоновка результирующих четвертей
    # C11 = M1 + M4 - M5 + M7
    c11 = add_pure(sub_pure(add_pure(m1, m4), m5), m7)
    # C12 = M3 + M5
    c12 = add_pure(m3, m5)
    # C21 = M2 + M4
    c21 = add_pure(m2, m4)
    # C22 = M1 - M2 + M3 + M6
    c22 = add_pure(sub_pure(add_pure(m1, m3), m2), m6)

    # Сборка итоговой матрицы из четвертей
    res = []
    for i in range(mid):
        res.append(c11[i] + c12[i])
    for i in range(mid):
        res.append(c21[i] + c22[i])
        
    return res