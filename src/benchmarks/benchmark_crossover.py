import os
import csv
import time
import ctypes
from datetime import datetime
import numpy as np
from scipy.signal import correlate
from numpy.ctypeslib import ndpointer

# Предполагаемые импорты из вашей структуры проекта
try:
    from src.benchmarks.scenarios import get_predefined_scenarios
    from src.amplifier.models import power_amplifier_model
    from src.amplifier.feedback_path import channel_emulation
    from src.signals.generation import simulate_ofdm_frame
    from src.signals.metrics import calculate_evm, calculate_aclr, calculate_papr
except ImportError:
    # Заглушки на случай автономного запуска вне структуры проекта
    def get_predefined_scenarios():
        return {
            'test_scenario': {
                'n_frames': 1, 'n_fft': 1024, 'cp_len': 72, 'n_active': 600,
                'backoff': 0.89, 'fb_snr': 50, 'reg': 1e-6,
                'pa_params': {'p_sat': 10, 'gain_db': 20}
            }
        }
    def simulate_ofdm_frame(*args): return np.random.randn(10000) + 1j*np.random.randn(10000), 122.88e6, 20e6
    def oversample_fft(sig, factor): return np.repeat(sig, factor) # упрощено
    def power_amplifier_model(sig, **kwargs): return sig * 1.2
    def channel_emulation(sig, **kwargs): return sig + 1e-4*np.random.randn(len(sig))

# ----------------------------------------------------------------------
# Загрузка C-библиотеки и описание типов аргументов
# ----------------------------------------------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))

lib_name = 'libdpd_riscv.dll' if os.name == 'nt' else 'libdpd_riscv.so'
lib_path = os.path.join(project_root, lib_name)
lib = ctypes.CDLL(lib_path)

c_cmplx_2d = ndpointer(dtype=np.complex128, ndim=2, flags='C_CONTIGUOUS')
c_cmplx_1d = ndpointer(dtype=np.complex128, ndim=1, flags='C_CONTIGUOUS')
c_double_1d = ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS')

# Передаем b_s в качестве 3-го параметра (тип ctypes.c_int) согласно вашей C-библиотеке
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

lib.solve_qr_householder.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, c_cmplx_2d, c_cmplx_1d, ctypes.c_double,
                                     c_cmplx_1d, c_cmplx_2d, c_cmplx_1d, c_cmplx_1d]

# ----------------------------------------------------------------------
# Функция-обертка над C-методами с динамическим b_s
# ----------------------------------------------------------------------
def wrap_solver(c_func, solver_type):
    def solver(A, y, reg):
        n_s, n_f = A.shape
        
        # Индивидуальный подбор b_s на основе результатов вашего свитч-теста
        if 'Winograd' in solver_type:
            b_s = 32  # Пик производительности для Винограда (57.2 ms)
        elif 'Perminov' in solver_type:
            b_s = 16  # Строго 16! Выше начинается жесткая деградация памяти (74.2 ms)
        else:
            b_s = 32  # Оптимальный баланс для Naive-методов и QR
        
        w_out = np.zeros(n_f, dtype=np.complex128)
        A_c = np.ascontiguousarray(A, dtype=np.complex128)
        y_c = np.ascontiguousarray(y, dtype=np.complex128)
        
        # Динамическая аллокация буферов под выверенный b_s
        Ah = np.zeros((n_f, b_s), dtype=np.complex128) 
        Ah_A = np.zeros((n_f, n_f), dtype=np.complex128)
        Ah_y = np.zeros(n_f, dtype=np.complex128)
        BT = np.zeros((n_f, b_s), dtype=np.complex128)

        if solver_type == 'Gauss_Naive':
            M = np.zeros((n_f, n_f + 1), dtype=np.complex128)
            start_time = time.perf_counter()
            c_func(n_s, n_f, b_s, A_c, y_c, ctypes.c_double(reg), w_out, Ah, Ah_A, Ah_y, M, BT)
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        elif solver_type == 'Gauss_Perminov':
            M = np.zeros((n_f, n_f + 1), dtype=np.complex128)
            tA = np.zeros(n_f * b_s, dtype=np.float64)
            tBT = np.zeros(n_f * b_s, dtype=np.float64)
            start_time = time.perf_counter()
            c_func(n_s, n_f, b_s, A_c, y_c, ctypes.c_double(reg), w_out, Ah, Ah_A, Ah_y, M, tA, BT, tBT)
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        elif solver_type == 'Gauss_Winograd':
            M = np.zeros((n_f, n_f + 1), dtype=np.complex128)
            rf = np.zeros(n_f, dtype=np.complex128)
            cf = np.zeros(n_f, dtype=np.complex128)
            start_time = time.perf_counter()
            c_func(n_s, n_f, b_s, A_c, y_c, ctypes.c_double(reg), w_out, Ah, Ah_A, Ah_y, M, rf, cf, BT)
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        elif solver_type == 'Cholesky_Naive':
            start_time = time.perf_counter()
            c_func(n_s, n_f, b_s, A_c, y_c, ctypes.c_double(reg), w_out, Ah, Ah_A, Ah_y, BT)
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        elif solver_type == 'Cholesky_Perminov':
            tA = np.zeros(n_f * b_s, dtype=np.float64)
            tBT = np.zeros(n_f * b_s, dtype=np.float64)
            start_time = time.perf_counter()
            c_func(n_s, n_f, b_s, A_c, y_c, ctypes.c_double(reg), w_out, Ah, Ah_A, Ah_y, tA, BT, tBT)
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        elif solver_type == 'Cholesky_Winograd':
            rf = np.zeros(n_f, dtype=np.complex128)
            cf = np.zeros(n_f, dtype=np.complex128)
            start_time = time.perf_counter()
            c_func(n_s, n_f, b_s, A_c, y_c, ctypes.c_double(reg), w_out, Ah, Ah_A, Ah_y, rf, cf, BT)
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        elif solver_type == 'QR_Householder':
            R = np.zeros((n_s + n_f, n_f), dtype=np.complex128)
            y_ext = np.zeros(n_s + n_f, dtype=np.complex128)
            v = np.zeros(n_s + n_f, dtype=np.complex128)
            start_time = time.perf_counter()
            c_func(n_s, n_f, b_s, A_c, y_c, ctypes.c_double(reg), w_out, R, y_ext, v)
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        return w_out, elapsed_ms
    return solver

solvers = {
    'Gauss_Naive':       wrap_solver(lib.solve_gauss_naive,      'Gauss_Naive'),
    'Gauss_Perminov':   wrap_solver(lib.solve_gauss_perminov,   'Gauss_Perminov'),
    'Gauss_Winograd':   wrap_solver(lib.solve_gauss_winograd,   'Gauss_Winograd'),
    'Cholesky_Naive':   wrap_solver(lib.solve_cholesky_naive,   'Cholesky_Naive'),
    'Cholesky_Perminov':wrap_solver(lib.solve_cholesky_perminov,'Cholesky_Perminov'),
    'Cholesky_Winograd':wrap_solver(lib.solve_cholesky_winograd,'Cholesky_Winograd')
}

# ----------------------------------------------------------------------
# Вспомогательные функции ЦОС и GMP-модели
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

def generate_training_data(cfg, train_len=4096):
    os_factor = 4
    x_raw, fs, bw = simulate_ofdm_frame(cfg['n_frames'], cfg['n_fft'], cfg['cp_len'], cfg['n_active'])
    x_os = oversample_fft(x_raw, os_factor)
    x_max = np.max(np.abs(x_os))
    backoff = cfg.get('backoff', 0.89)
    x_target_norm = (x_os / x_max) * backoff

    y_pa = power_amplifier_model(x_os, mode='AB', **cfg['pa_params'])
    y_fb = channel_emulation(y_pa, snr_db=cfg.get('fb_snr', 50))
    y_sync = align_signals_strictly(x_os, y_fb)

    y_train = y_sync[:train_len]
    phase_corr = np.exp(-1j * np.angle(np.sum(y_train * np.conj(x_target_norm[:train_len]))))
    y_train_norm = (y_train * phase_corr) / np.max(np.abs(y_train))

    target = x_target_norm[:train_len]
    return y_train_norm, target, fs, bw, x_os, x_target_norm, backoff, x_max, os_factor

# ----------------------------------------------------------------------
# Функция запуска кроссовер-бенчмарка (Оптимизированная по числу итераций)
# ----------------------------------------------------------------------
def run_crossover_benchmark():
    scenarios = get_predefined_scenarios()
    train_len = 4096
    csv_file = 'crossover_results.csv'
    file_exists = os.path.isfile(csv_file)

    # 1. Репрезентативный прогрев системы с использованием оптимального b_s = 32
    print("Быстрый прогрев системы (кэш-память и стабилизация CPU)...")
    dummy_A = np.random.randn(512, 32) + 1j * np.random.randn(512, 32)
    dummy_y = np.random.randn(512) + 1j * np.random.randn(512)
    for _ in range(3):
        _ = solvers['Gauss_Winograd'](dummy_A, dummy_y, 1e-6)

    # Заданная сетка параметров полинома GMP
    param_grid = []
    for p in [3, 5, 7, 9, 11]:
        for m in [2, 3, 4, 5]:
            for c in [0, 1, 2]:
                param_grid.append({'p_order': p, 'm_depth': m, 'cross_depth': c})

    with open(csv_file, mode='a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['Scenario', 'Solver', 'n_features', 'n_samples', 'Time_ms', 'RelError', 'Timestamp'])

        for scen_name, cfg in scenarios.items():
            print(f"\n=== Сценарий: {scen_name} ===")
            y_norm, target, _, _, _, _, _, _, _ = generate_training_data(cfg, train_len)

            for params in param_grid:
                A_train = create_gmp_matrix(y_norm, **params)
                n_samples, n_features = A_train.shape
                
                if n_features < 5: 
                    continue  

                reg = cfg.get('reg', 1e-6)

                for solver_name, solver_func in solvers.items():
                    times = []
                    
                    # 2. Минимально репрезентативный статистический цикл (5 повторений вместо 50)
                    for _ in range(5):
                        _, elapsed = solver_func(A_train, target, reg)
                        times.append(elapsed)
                    
                    # Медиана по 5 проходам надежно отсечет случайный единичный выброс от ОС
                    median_time = np.median(times)
                    
                    # 3. Расчет точности решения
                    w, _ = solver_func(A_train, target, reg)
                    residual = A_train @ w - target
                    rel_err = np.linalg.norm(residual) / np.linalg.norm(target)

                    writer.writerow([
                        scen_name, solver_name, n_features, n_samples,
                        round(median_time, 4), f"{rel_err:.6e}", datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    ])
                    f.flush()
                    print(f"  {solver_name:20} | n_feat={n_features:4d} | time={median_time:8.3f} ms | err={rel_err:.2e}")

            print("-" * 80)

if __name__ == "__main__":
    np.random.seed(42)
    run_crossover_benchmark()