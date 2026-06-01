"""
crossover_benchmark.py
Исследование зависимости времени и точности линейных солверов от размера матрицы
в контексте DPD. Для каждого сценария генерируется сигнал, формируется обучающая
выборка фиксированного размера (4096 отсчётов), затем варьируется число признаков
(p_order, m_depth, cross_depth), чтобы получить разные n_features. Измеряется время
решения линейной системы (A^H A + reg I) w = A^H y и относительная ошибка.
"""

import os
import csv
import time
import ctypes
import numpy as np
from datetime import datetime
from scipy.signal import correlate
from numpy.ctypeslib import ndpointer

from src.benchmarks.scenarios import get_predefined_scenarios
from src.amplifier.models import power_amplifier_model
from src.amplifier.feedback_path import channel_emulation
from src.signals.generation import simulate_ofdm_frame
from src.signals.metrics import calculate_evm, calculate_aclr, calculate_papr

# ----------------------------------------------------------------------
# Загрузка C-библиотеки и обёртки 
# ----------------------------------------------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))

lib_name = 'libdpd_riscv.dll' if os.name == 'nt' else 'libdpd_riscv.so'
lib_path = os.path.join(project_root, lib_name)
lib = ctypes.CDLL(lib_path)

c_cmplx_2d = ndpointer(dtype=np.complex128, ndim=2, flags='C_CONTIGUOUS')
c_cmplx_1d = ndpointer(dtype=np.complex128, ndim=1, flags='C_CONTIGUOUS')
c_double_1d = ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS')

lib.solve_gauss_naive.argtypes = [ctypes.c_int, ctypes.c_int, c_cmplx_2d, c_cmplx_1d, ctypes.c_double,
                                  c_cmplx_1d, c_cmplx_2d, c_cmplx_2d, c_cmplx_1d, c_cmplx_2d, c_cmplx_2d]
lib.solve_gauss_perminov.argtypes = [ctypes.c_int, ctypes.c_int, c_cmplx_2d, c_cmplx_1d, ctypes.c_double,
                                     c_cmplx_1d, c_cmplx_2d, c_cmplx_2d, c_cmplx_1d, c_cmplx_2d,
                                     c_double_1d, c_cmplx_2d, c_double_1d]
lib.solve_gauss_winograd.argtypes = [ctypes.c_int, ctypes.c_int, c_cmplx_2d, c_cmplx_1d, ctypes.c_double,
                                     c_cmplx_1d, c_cmplx_2d, c_cmplx_2d, c_cmplx_1d, c_cmplx_2d,
                                     c_cmplx_1d, c_cmplx_1d, c_cmplx_2d]
lib.solve_cholesky_naive.argtypes = [ctypes.c_int, ctypes.c_int, c_cmplx_2d, c_cmplx_1d, ctypes.c_double,
                                     c_cmplx_1d, c_cmplx_2d, c_cmplx_2d, c_cmplx_1d, c_cmplx_2d]
lib.solve_cholesky_perminov.argtypes = [ctypes.c_int, ctypes.c_int, c_cmplx_2d, c_cmplx_1d, ctypes.c_double,
                                        c_cmplx_1d, c_cmplx_2d, c_cmplx_2d, c_cmplx_1d, c_double_1d, c_cmplx_2d, c_double_1d]
lib.solve_cholesky_winograd.argtypes = [ctypes.c_int, ctypes.c_int, c_cmplx_2d, c_cmplx_1d, ctypes.c_double,
                                        c_cmplx_1d, c_cmplx_2d, c_cmplx_2d, c_cmplx_1d, c_cmplx_1d, c_cmplx_1d, c_cmplx_2d]
lib.solve_qr_householder.argtypes = [ctypes.c_int, ctypes.c_int, c_cmplx_2d, c_cmplx_1d, ctypes.c_double,
                                     c_cmplx_1d, c_cmplx_2d, c_cmplx_1d, c_cmplx_1d]

def wrap_solver(c_func, solver_type):
    def solver(A, y, reg):
        n_s, n_f = A.shape
        w_out = np.zeros(n_f, dtype=np.complex128)
        A_c = np.ascontiguousarray(A, dtype=np.complex128)
        y_c = np.ascontiguousarray(y, dtype=np.complex128)
        Ah = np.zeros((n_f, n_s), dtype=np.complex128)
        Ah_A = np.zeros((n_f, n_f), dtype=np.complex128)
        Ah_y = np.zeros(n_f, dtype=np.complex128)
        BT = np.zeros((n_f, n_s), dtype=np.complex128)

        # Выносим аллокации за пределы таймера для точного измерения работы С-кода
        if solver_type == 'Gauss_Naive':
            M = np.zeros((n_f, n_f + 1), dtype=np.complex128)
            start_time = time.perf_counter()
            c_func(n_s, n_f, A_c, y_c, ctypes.c_double(reg), w_out, Ah, Ah_A, Ah_y, M, BT)
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        elif solver_type == 'Gauss_Perminov':
            M = np.zeros((n_f, n_f + 1), dtype=np.complex128)
            tA = np.zeros(n_f * n_s, dtype=np.float64)
            tBT = np.zeros(n_f * n_s, dtype=np.float64)
            start_time = time.perf_counter()
            c_func(n_s, n_f, A_c, y_c, ctypes.c_double(reg), w_out, Ah, Ah_A, Ah_y, M, tA, BT, tBT)
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        elif solver_type == 'Gauss_Winograd':
            M = np.zeros((n_f, n_f + 1), dtype=np.complex128)
            rf = np.zeros(n_f, dtype=np.complex128)
            cf = np.zeros(n_f, dtype=np.complex128)
            start_time = time.perf_counter()
            c_func(n_s, n_f, A_c, y_c, ctypes.c_double(reg), w_out, Ah, Ah_A, Ah_y, M, rf, cf, BT)
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        elif solver_type == 'Cholesky_Naive':
            start_time = time.perf_counter()
            c_func(n_s, n_f, A_c, y_c, ctypes.c_double(reg), w_out, Ah, Ah_A, Ah_y, BT)
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        elif solver_type == 'Cholesky_Perminov':
            tA = np.zeros(n_f * n_s, dtype=np.float64)
            tBT = np.zeros(n_f * n_s, dtype=np.float64)
            start_time = time.perf_counter()
            c_func(n_s, n_f, A_c, y_c, ctypes.c_double(reg), w_out, Ah, Ah_A, Ah_y, tA, BT, tBT)
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        elif solver_type == 'Cholesky_Winograd':
            rf = np.zeros(n_f, dtype=np.complex128)
            cf = np.zeros(n_f, dtype=np.complex128)
            start_time = time.perf_counter()
            c_func(n_s, n_f, A_c, y_c, ctypes.c_double(reg), w_out, Ah, Ah_A, Ah_y, rf, cf, BT)
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        elif solver_type == 'QR_Householder':
            R = np.zeros((n_s + n_f, n_f), dtype=np.complex128)
            y_ext = np.zeros(n_s + n_f, dtype=np.complex128)
            v = np.zeros(n_s + n_f, dtype=np.complex128)
            start_time = time.perf_counter()
            c_func(n_s, n_f, A_c, y_c, ctypes.c_double(reg), w_out, R, y_ext, v)
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        return w_out, elapsed_ms
    return solver

solvers = {
    'Gauss_Naive':      wrap_solver(lib.solve_gauss_naive,      'Gauss_Naive'),
    'Gauss_Perminov':   wrap_solver(lib.solve_gauss_perminov,   'Gauss_Perminov'),
    'Gauss_Winograd':   wrap_solver(lib.solve_gauss_winograd,   'Gauss_Winograd'),
    'Cholesky_Naive':   wrap_solver(lib.solve_cholesky_naive,   'Cholesky_Naive'),
    'Cholesky_Perminov':wrap_solver(lib.solve_cholesky_perminov,'Cholesky_Perminov'),
    'Cholesky_Winograd':wrap_solver(lib.solve_cholesky_winograd,'Cholesky_Winograd')
}

# ----------------------------------------------------------------------
# Вспомогательные функции ЦОС (как в основном коде)
# ----------------------------------------------------------------------
def align_signals_strictly(ref, meas):
    c = correlate(ref, meas, method='fft', mode='full')
    s = np.argmax(np.abs(c)) - (len(meas) - 1)
    return np.roll(meas, s)

def oversample_fft(signal, os_factor):
    n = len(signal)
    n_os = n * os_factor
    spec = np.fft.fft(signal)
    half = (n + 1) // 2
    spec_os = np.zeros(n_os, dtype=complex)
    spec_os[:half] = spec[:half]
    spec_os[-(n - half):] = spec[half:]
    return np.fft.ifft(spec_os) * os_factor

def create_gmp_matrix(x, p_order=3, m_depth=2, cross_depth=1):
    x = np.asarray(x)
    n, abs_x = len(x), np.abs(x)
    matrix_columns = []
    powers = np.arange(1, p_order + 1, 2)
    for m in range(m_depth):
        x_del, abs_del = np.zeros(n, dtype=complex), np.zeros(n, dtype=float)
        x_del[m:] = x[:n - m]
        abs_del[m:] = abs_x[:n - m]
        for p in powers:
            matrix_columns.append(x_del * (abs_del ** (p - 1)))
        if cross_depth > 0:
            for l in range(1, cross_depth + 1):
                if m + l < n:
                    abs_lag = np.zeros(n, dtype=float)
                    abs_lag[m + l:] = abs_x[:n - (m + l)]
                    for p in powers[1:]:
                        matrix_columns.append(x_del * (abs_lag ** (p - 1)))
    return np.column_stack(matrix_columns)

# ----------------------------------------------------------------------
# Генерация обучающей пары (A_train, y_train) для заданного сценария
# ----------------------------------------------------------------------
def generate_training_data(cfg, train_len=4096):
    os_factor = 4
    x_raw, fs, bw = simulate_ofdm_frame(cfg['n_frames'], cfg['n_fft'], cfg['cp_len'], cfg['n_active'])
    x_os = oversample_fft(x_raw, os_factor)
    x_max = np.max(np.abs(x_os))
    backoff = cfg.get('backoff', 0.89)
    x_target_norm = (x_os / x_max) * backoff

    # Один проход PA без DPD для получения y_fb
    y_pa = power_amplifier_model(x_os, mode='AB', **cfg['pa_params'])
    y_fb = channel_emulation(y_pa, snr_db=cfg.get('fb_snr', 50))
    y_sync = align_signals_strictly(x_os, y_fb)

    # Нормировка y_train
    y_train = y_sync[:train_len]
    phase_corr = np.exp(-1j * np.angle(np.sum(y_train * np.conj(x_target_norm[:train_len]))))
    y_train_norm = (y_train * phase_corr) / np.max(np.abs(y_train))

    # Целевой вектор
    target = x_target_norm[:train_len]

    return y_train_norm, target, fs, bw, x_os, x_target_norm, backoff, x_max, os_factor

# ----------------------------------------------------------------------
# Основная функция кроссовер-бенчмарка
# ----------------------------------------------------------------------
def run_crossover_benchmark():
    scenarios = get_predefined_scenarios()
    train_len = 4096
    csv_file = 'crossover_results.csv'
    file_exists = os.path.isfile(csv_file)

    with open(csv_file, mode='a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['Scenario', 'Solver', 'n_features', 'n_samples', 'Time_ms', 'RelError', 'Timestamp'])

        # Сетка параметров: расширена для тестирования до ~120 признаков
        param_grid = []
        for p in [3, 5, 7, 9, 11]:        # p_order (всегда берем нечетные порядки, как это принято)
            for m in [2, 3, 4, 5]:           # m_depth (добавляем бóльшую глубину памяти)
                for c in [0, 1, 2]:       # cross_depth
                    param_grid.append({'p_order': p, 'm_depth': m, 'cross_depth': c})

        for scen_name, cfg in scenarios.items():
            print(f"\n=== Сценарий: {scen_name} ===")
            y_norm, target, fs, bw, x_os, x_target_norm, backoff, x_max, os_factor = generate_training_data(cfg, train_len)

            for params in param_grid:
                p_order = params['p_order']
                m_depth = params['m_depth']
                cross_depth = params['cross_depth']

                A_train = create_gmp_matrix(y_norm, p_order, m_depth, cross_depth)
                n_samples, n_features = A_train.shape
                
                # Защита от слишком маленьких или слишком огромных матриц
                if n_features < 5: 
                    continue  

                reg = cfg.get('reg', 1e-6)

                # Усреднение увеличено до 10 для сглаживания ОС-джиттера
                n_repeats = 5
                for solver_name, solver_func in solvers.items():
                    # Warm-up (прогрев кеша) для честного замера
                    _ = solver_func(A_train, target, reg)
                    
                    times = []
                    errors = []
                    for _ in range(n_repeats):
                        w, elapsed = solver_func(A_train, target, reg)
                        
                        residual = A_train @ w - target
                        rel_err = np.linalg.norm(residual) / np.linalg.norm(target)
                        
                        times.append(elapsed)
                        errors.append(rel_err)

                    avg_time = np.mean(times)
                    avg_err = np.mean(errors)

                    writer.writerow([scen_name, solver_name, n_features, n_samples,
                                     round(avg_time, 4), f"{avg_err:.6e}",
                                     datetime.now().strftime("%Y-%m-%d %H:%M:%S")])
                    f.flush()
                    print(f"  {solver_name:20} | n_feat={n_features:4d} | time={avg_time:8.2f} ms | rel_err={avg_err:.2e}")

            print("-" * 80)

if __name__ == "__main__":
    np.random.seed(42)
    run_crossover_benchmark()