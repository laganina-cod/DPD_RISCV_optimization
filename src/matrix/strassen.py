import numpy as np
from .utils import pad_to_power_of_two, trim_matrix

def _strassen_rec_ultra(a, b, c, leaf_size, workspace):
    n = a.shape[0]
    if n <= leaf_size:
        np.matmul(a, b, out=c)
        return

    mid = n // 2
    size = mid * mid
    
    # 7 матриц P и 2 временных буфера T
    p_mats = [workspace[i*size : (i+1)*size].reshape(mid, mid) for i in range(7)]
    t1 = workspace[7*size : 8*size].reshape(mid, mid)
    t2 = workspace[8*size : 9*size].reshape(mid, mid)
    workspace_next = workspace[9*size:]

    a11, a12, a21, a22 = a[:mid, :mid], a[:mid, mid:], a[mid:, :mid], a[mid:, mid:]
    b11, b12, b21, b22 = b[:mid, :mid], b[:mid, mid:], b[mid:, :mid], b[mid:, mid:]
    c11, c12, c21, c22 = c[:mid, :mid], c[:mid, mid:], c[mid:, :mid], c[mid:, mid:]

    # P1..P7 (Рекурсия)
    np.add(a11, a22, out=t1); np.add(b11, b22, out=t2)
    _strassen_rec_ultra(t1, t2, p_mats[0], leaf_size, workspace_next) # P1
    np.add(a21, a22, out=t1)
    _strassen_rec_ultra(t1, b11, p_mats[1], leaf_size, workspace_next) # P2
    np.subtract(b12, b22, out=t2)
    _strassen_rec_ultra(a11, t2, p_mats[2], leaf_size, workspace_next) # P3
    np.subtract(b21, b11, out=t2)
    _strassen_rec_ultra(a22, t2, p_mats[3], leaf_size, workspace_next) # P4
    np.add(a11, a12, out=t1)
    _strassen_rec_ultra(t1, b22, p_mats[4], leaf_size, workspace_next) # P5
    np.subtract(a21, a11, out=t1); np.add(b11, b12, out=t2)
    _strassen_rec_ultra(t1, t2, p_mats[5], leaf_size, workspace_next) # P6
    np.subtract(a12, a22, out=t1); np.add(b21, b22, out=t2)
    _strassen_rec_ultra(t1, t2, p_mats[6], leaf_size, workspace_next) # P7

    # Сборка C
    np.add(p_mats[0], p_mats[3], out=c11)
    np.subtract(c11, p_mats[4], out=c11)
    np.add(c11, p_mats[6], out=c11) # C11
    np.add(p_mats[2], p_mats[4], out=c12) # C12
    np.add(p_mats[1], p_mats[3], out=c21) # C21
    np.subtract(p_mats[0], p_mats[1], out=c22)
    np.add(c22, p_mats[2], out=c22)
    np.add(c22, p_mats[5], out=c22) # C22


def matmul_strassen(a, b, leaf_size=64, max_block=2048):
    """
    Доработанная версия алгоритма Штрассена с блочным разбиением (Tiling).
    Исключает избыточный паддинг прямоугольных матриц и OOM-ошибки.
    """
    R, K = a.shape
    K_b, C = b.shape
    assert K == K_b, "Несовпадение внутренних размерностей матриц!"

    # Задаем безопасные границы подблоков
    R_block = min(R, max_block)
    K_block = min(K, max_block)
    C_block = min(C, max_block)

    # Определяем максимально возможную степень двойки для аллокации единого буфера
    max_dim = max(R_block, K_block, C_block)
    n_padded = 1 if max_dim == 0 else 2 ** int(np.ceil(np.log2(max_dim)))
    
    # Выделяем память ОДИН раз под максимальный размер блока
    workspace = np.empty(5 * n_padded * n_padded, dtype=a.dtype)
    cp_buffer = np.empty((n_padded, n_padded), dtype=a.dtype)
    
    # Финальная результирующая матрица
    res = np.zeros((R, C), dtype=a.dtype)

    # Итерируемся блоками по внешним и внутренним размерностям
    for i in range(0, R, R_block):
        i_end = min(i + R_block, R)
        for j in range(0, C, C_block):
            j_end = min(j + C_block, C)
            
            # Аккумулятор для текущего вычисленного суб-блока матрицы C
            acc_block = np.zeros((i_end - i, j_end - j), dtype=a.dtype)
            
            for e in range(0, K, K_block):
                e_end = min(e + K_block, K)
                
                # Выделяем прямоугольные слайсы
                a_slice = a[i:i_end, e:e_end]
                b_slice = b[e:e_end, j:j_end]
                
                # Достраиваем до квадрата степени двойки локально
                ap, bp, sub_rows, sub_cols = pad_to_power_of_two(a_slice, b_slice)
                
                n_sub = ap.shape[0]
                # Нарезаем вьюхи из ранее созданных глобальных буферов
                sub_workspace = workspace[:5 * n_sub * n_sub]
                cp = cp_buffer[:n_sub, :n_sub]
                cp.fill(0)  # Сбрасываем старые значения в буфере
                
                # Запуск стабильного Штрассена без риска сожрать всю оперативку
                _strassen_rec_ultra(ap, bp, cp, leaf_size, sub_workspace)
                
                # Обрезаем паддинг и аккумулируем сумму по внутренней размерности K
                acc_block += trim_matrix(cp, sub_rows, sub_cols)
            
            # Записываем накопленный блок в итоговую матрицу
            res[i:i_end, j:j_end] = acc_block
            
    return res