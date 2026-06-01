import ctypes
import numpy as np
import time
import csv                     # <-- добавлен импорт для CSV

# Загрузка библиотеки
lib = ctypes.CDLL("./libdpd_riscv.so")

# Настройка типов данных для solve_gauss_winograd в качестве примера
lib.solve_gauss_perminov.argtypes = [
    ctypes.c_int, ctypes.c_int, ctypes.c_int,  # n_s, n_f, b_s
    ctypes.c_void_p, ctypes.c_void_p,          # A, y
    ctypes.c_double, ctypes.c_void_p,          # reg, w
    ctypes.c_void_p, ctypes.c_void_p,          # Ah, Ah_A
    ctypes.c_void_p, ctypes.c_void_p,          # Ah_y, M
    ctypes.c_void_p, ctypes.c_void_p,          # rf, cf
    ctypes.c_void_p                            # BT
]

# Параметры теста
Ns = 16000
Nf = 32
reg = 1e-4

# Генерация синтетических данных
A = (np.random.randn(Ns, Nf) + 1j * np.random.randn(Ns, Nf)).astype(np.complex128)
y = (np.random.randn(Ns) + 1j * np.random.randn(Ns)).astype(np.complex128)
w = np.zeros(Nf, dtype=np.complex128)

# Выделение рабочих массивов (выделяются под максимальный размер блока)
Ah = np.zeros(Nf * Ns, dtype=np.complex128) 
Ah_A = np.zeros((Nf, Nf), dtype=np.complex128)
Ah_y = np.zeros(Nf, dtype=np.complex128)
M = np.zeros((Nf, Nf + 1), dtype=np.complex128)
rf = np.zeros(Nf, dtype=np.complex128)
cf = np.zeros(Nf, dtype=np.complex128)
BT = np.zeros(Nf * Ns, dtype=np.complex128)

# Сетка размеров блоков
block_sizes = [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, Ns]

# Открываем CSV-файл для записи (перезапись при каждом запуске)
with open('gauss_perminov_benchmark.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['BlockSize', 'Time_ms'])   # заголовок

    print(f"Размер блока Bs | Время выполнения (мс)")
    print("-" * 40)

    for bs in block_sizes:
        # Разогрев кэша
        lib.solve_gauss_perminov(Ns, Nf, bs, A.ctypes.data, y.ctypes.data, reg, w.ctypes.data,
                                 Ah.ctypes.data, Ah_A.ctypes.data, Ah_y.ctypes.data, M.ctypes.data,
                                 rf.ctypes.data, cf.ctypes.data, BT.ctypes.data)
        
        # Итерационный замер времени
        iters = 20
        t0 = time.perf_counter_ns()
        for _ in range(iters):
            lib.solve_gauss_perminov(Ns, Nf, bs, A.ctypes.data, y.ctypes.data, reg, w.ctypes.data,
                                     Ah.ctypes.data, Ah_A.ctypes.data, Ah_y.ctypes.data, M.ctypes.data,
                                     rf.ctypes.data, cf.ctypes.data, BT.ctypes.data)
        t1 = time.perf_counter_ns()
        
        avg_time_ms = ((t1 - t0) / iters) / 1e6
        print(f"{bs:15d} | {avg_time_ms:.4f}")
        writer.writerow([bs, avg_time_ms])      # <-- запись строки в CSV