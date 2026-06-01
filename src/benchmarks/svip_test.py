import os
import csv
import time
import ctypes
import numpy as np
from numpy.ctypeslib import ndpointer

def run_blocksize_benchmark():
    # Загрузка C-библиотеки
    current_dir = os.path.dirname(os.path.abspath(__file__))
    lib_name = 'libdpd_riscv.dll' if os.name == 'nt' else 'libdpd_riscv.so'
    lib_path = os.path.join(current_dir, lib_name)
    
    try:
        lib = ctypes.CDLL(lib_path)
    except OSError:
        print(f"Ошибка: Не удалось загрузить библиотеку {lib_path}")
        return

    # Настройка типов данных
    c_cmplx_2d = ndpointer(dtype=np.complex128, ndim=2, flags='C_CONTIGUOUS')
    c_cmplx_1d = ndpointer(dtype=np.complex128, ndim=1, flags='C_CONTIGUOUS')
    c_double_1d = ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS')

    # Сигнатуры с добавленным параметром b_s (int)
    lib.solve_gauss_naive.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, c_cmplx_2d, c_cmplx_1d, ctypes.c_double,
                                      c_cmplx_1d, c_cmplx_2d, c_cmplx_2d, c_cmplx_1d, c_cmplx_2d, c_cmplx_2d]
    lib.solve_gauss_perminov.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, c_cmplx_2d, c_cmplx_1d, ctypes.c_double,
                                         c_cmplx_1d, c_cmplx_2d, c_cmplx_2d, c_cmplx_1d, c_cmplx_2d,
                                         c_double_1d, c_cmplx_2d, c_double_1d]
    lib.solve_gauss_winograd.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, c_cmplx_2d, c_cmplx_1d, ctypes.c_double,
                                         c_cmplx_1d, c_cmplx_2d, c_cmplx_2d, c_cmplx_1d, c_cmplx_2d,
                                         c_cmplx_1d, c_cmplx_1d, c_cmplx_2d]
    lib.solve_cholesky_naive.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, c_cmplx_2d, c_cmplx_1d, ctypes.c_double,
                                         c_cmplx_1d, c_cmplx_2d, c_cmplx_2d, c_cmplx_1d, c_cmplx_2d]
    lib.solve_cholesky_perminov.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, c_cmplx_2d, c_cmplx_1d, ctypes.c_double,
                                            c_cmplx_1d, c_cmplx_2d, c_cmplx_2d, c_cmplx_1d, c_double_1d, c_cmplx_2d, c_double_1d]
    lib.solve_cholesky_winograd.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, c_cmplx_2d, c_cmplx_1d, ctypes.c_double,
                                            c_cmplx_1d, c_cmplx_2d, c_cmplx_2d, c_cmplx_1d, c_cmplx_1d, c_cmplx_1d, c_cmplx_2d]

    # Универсальная обертка для каждого решателя с изоляцией аллокаций памяти
    def wrap_solver(c_func, solver_type):
        def solver(Ns, Nf, bs, A, y, reg, w):
            Ah = np.zeros((Nf, Ns), dtype=np.complex128)
            Ah_A = np.zeros((Nf, Nf), dtype=np.complex128)
            Ah_y = np.zeros(Nf, dtype=np.complex128)
            BT = np.zeros((Nf, Ns), dtype=np.complex128)

            if solver_type == 'Gauss_Naive':
                M = np.zeros((Nf, Nf + 1), dtype=np.complex128)
                t0 = time.perf_counter_ns()
                c_func(Ns, Nf, bs, A, y, reg, w, Ah, Ah_A, Ah_y, M, BT)
                t1 = time.perf_counter_ns()

            elif solver_type == 'Gauss_Perminov':
                M = np.zeros((Nf, Nf + 1), dtype=np.complex128)
                tA = np.zeros(Nf * Ns, dtype=np.float64)
                tBT = np.zeros(Nf * Ns, dtype=np.float64)
                t0 = time.perf_counter_ns()
                c_func(Ns, Nf, bs, A, y, reg, w, Ah, Ah_A, Ah_y, M, tA, BT, tBT)
                t1 = time.perf_counter_ns()

            elif solver_type == 'Gauss_Winograd':
                M = np.zeros((Nf, Nf + 1), dtype=np.complex128)
                rf = np.zeros(Nf, dtype=np.complex128)
                cf = np.zeros(Nf, dtype=np.complex128)
                t0 = time.perf_counter_ns()
                c_func(Ns, Nf, bs, A, y, reg, w, Ah, Ah_A, Ah_y, M, rf, cf, BT)
                t1 = time.perf_counter_ns()

            elif solver_type == 'Cholesky_Naive':
                t0 = time.perf_counter_ns()
                c_func(Ns, Nf, bs, A, y, reg, w, Ah, Ah_A, Ah_y, BT)
                t1 = time.perf_counter_ns()

            elif solver_type == 'Cholesky_Perminov':
                tA = np.zeros(Nf * Ns, dtype=np.float64)
                tBT = np.zeros(Nf * Ns, dtype=np.float64)
                t0 = time.perf_counter_ns()
                c_func(Ns, Nf, bs, A, y, reg, w, Ah, Ah_A, Ah_y, tA, BT, tBT)
                t1 = time.perf_counter_ns()

            elif solver_type == 'Cholesky_Winograd':
                rf = np.zeros(Nf, dtype=np.complex128)
                cf = np.zeros(Nf, dtype=np.complex128)
                t0 = time.perf_counter_ns()
                c_func(Ns, Nf, bs, A, y, reg, w, Ah, Ah_A, Ah_y, rf, cf, BT)
                t1 = time.perf_counter_ns()
            
            return (t1 - t0) / 1e6 # Конвертация наносекунд в миллисекунды
        return solver

    # Словарь со всеми тестируемыми методами
    solvers = {
        'Gauss_Naive':       wrap_solver(lib.solve_gauss_naive,       'Gauss_Naive'),
        'Gauss_Perminov':    wrap_solver(lib.solve_gauss_perminov,    'Gauss_Perminov'),
        'Gauss_Winograd':    wrap_solver(lib.solve_gauss_winograd,    'Gauss_Winograd'),
        'Cholesky_Naive':    wrap_solver(lib.solve_cholesky_naive,    'Cholesky_Naive'),
        'Cholesky_Perminov': wrap_solver(lib.solve_cholesky_perminov, 'Cholesky_Perminov'),
        'Cholesky_Winograd': wrap_solver(lib.solve_cholesky_winograd, 'Cholesky_Winograd')
    }

    # Характеристики сильно вытянутых матриц 
    Ns = 16000
    Nf = 32
    reg = 1e-4

    # Синтетические данные
    np.random.seed(42)
    A = np.random.randn(Ns, Nf) + 1j * np.random.randn(Ns, Nf)
    A = np.ascontiguousarray(A, dtype=np.complex128)
    y = np.random.randn(Ns) + 1j * np.random.randn(Ns)
    y = np.ascontiguousarray(y, dtype=np.complex128)
    
    # Сетка размеров блоков
    block_sizes = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, Ns]

    csv_file = 'blocksize_benchmark_all.csv'
    with open(csv_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Solver', 'BlockSize', 'Time_ms'])

        for solver_name, solver_func in solvers.items():
            print(f"\n=== Тестирование: {solver_name} ===")
            print(f"{'BlockSize':>10} | {'Time_ms':>10}")
            print("-" * 25)
            
            # Прогрев кэша (warm-up run)
            w_dummy = np.zeros(Nf, dtype=np.complex128)
            solver_func(Ns, Nf, 64, A, y, reg, w_dummy)
            
            for bs in block_sizes:
                w_out = np.zeros(Nf, dtype=np.complex128)
                
                # Итерационный замер для сглаживания ОС-джиттера
                iters = 10
                total_time_ms = 0
                for _ in range(iters):
                    total_time_ms += solver_func(Ns, Nf, bs, A, y, reg, w_out)
                
                avg_time_ms = total_time_ms / iters
                
                writer.writerow([solver_name, bs, round(avg_time_ms, 5)])
                print(f"{bs:10d} | {avg_time_ms:10.4f}")
                f.flush()

    print(f"\n[*] Все результаты сохранены в {csv_file}")

if __name__ == "__main__":
    run_blocksize_benchmark()