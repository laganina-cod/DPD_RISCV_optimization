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

# =====================================================================
# 1. Инициализация и загрузка C-библиотеки
# =====================================================================
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))

lib_name = 'libdpd_riscv.dll' if os.name == 'nt' else 'libdpd_riscv.so'
lib_path = os.path.join(project_root, lib_name)

try:
    lib = ctypes.CDLL(lib_path)
except OSError as e:
    raise RuntimeError(f"Не удалось загрузить {lib_name} по пути {lib_path}. {e}")

# Удобные псевдонимы для типов массивов
c_cmplx_2d = ndpointer(dtype=np.complex128, ndim=2, flags='C_CONTIGUOUS')
c_cmplx_1d = ndpointer(dtype=np.complex128, ndim=1, flags='C_CONTIGUOUS')
c_double_1d = ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS')

# --- Сигнатуры ---
lib.solve_gauss_naive.argtypes = [
    ctypes.c_int, ctypes.c_int, c_cmplx_2d, c_cmplx_1d, ctypes.c_double,
    c_cmplx_1d, c_cmplx_2d, c_cmplx_2d, c_cmplx_1d, c_cmplx_2d, c_cmplx_2d
]
lib.solve_gauss_perminov.argtypes = [
    ctypes.c_int, ctypes.c_int, c_cmplx_2d, c_cmplx_1d, ctypes.c_double,
    c_cmplx_1d, c_cmplx_2d, c_cmplx_2d, c_cmplx_1d, c_cmplx_2d,
    c_double_1d, c_cmplx_2d, c_double_1d
]
lib.solve_gauss_winograd.argtypes = [
    ctypes.c_int, ctypes.c_int, c_cmplx_2d, c_cmplx_1d, ctypes.c_double,
    c_cmplx_1d, c_cmplx_2d, c_cmplx_2d, c_cmplx_1d, c_cmplx_2d,
    c_cmplx_1d, c_cmplx_1d, c_cmplx_2d
]
lib.solve_cholesky_naive.argtypes = [
    ctypes.c_int, ctypes.c_int, c_cmplx_2d, c_cmplx_1d, ctypes.c_double,
    c_cmplx_1d, c_cmplx_2d, c_cmplx_2d, c_cmplx_1d, c_cmplx_2d
]
lib.solve_cholesky_perminov.argtypes = [
    ctypes.c_int, ctypes.c_int, c_cmplx_2d, c_cmplx_1d, ctypes.c_double,
    c_cmplx_1d, c_cmplx_2d, c_cmplx_2d, c_cmplx_1d, c_double_1d, c_cmplx_2d, c_double_1d
]
lib.solve_cholesky_winograd.argtypes = [
    ctypes.c_int, ctypes.c_int, c_cmplx_2d, c_cmplx_1d, ctypes.c_double,
    c_cmplx_1d, c_cmplx_2d, c_cmplx_2d, c_cmplx_1d, c_cmplx_1d, c_cmplx_1d, c_cmplx_2d
]
lib.solve_qr_householder.argtypes = [
    ctypes.c_int, ctypes.c_int, c_cmplx_2d, c_cmplx_1d, ctypes.c_double,
    c_cmplx_1d, c_cmplx_2d, c_cmplx_1d, c_cmplx_1d
]

# =====================================================================
# 2. Фабрика обёрток и вспомогательные функции
# =====================================================================

def align_signals_strictly(ref, meas):
    ref_norm = (ref - np.mean(ref)) / (np.std(ref) + 1e-9)
    meas_norm = (meas - np.mean(meas)) / (np.std(meas) + 1e-9)
    c = correlate(ref_norm, meas_norm, method='fft', mode='full')
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
    'Gauss_Naive':        wrap_solver(lib.solve_gauss_naive,        'Gauss_Naive'),
    'Gauss_Perminov':     wrap_solver(lib.solve_gauss_perminov,     'Gauss_Perminov'),
    'Gauss_Winograd':     wrap_solver(lib.solve_gauss_winograd,     'Gauss_Winograd'),
    'Cholesky_Naive':     wrap_solver(lib.solve_cholesky_naive,     'Cholesky_Naive'),
    'Cholesky_Perminov':  wrap_solver(lib.solve_cholesky_perminov,  'Cholesky_Perminov'),
    'Cholesky_Winograd':  wrap_solver(lib.solve_cholesky_winograd,  'Cholesky_Winograd'),
    'QR_Householder':     wrap_solver(lib.solve_qr_householder,     'QR_Householder')
}

# =====================================================================
# 3. Основной цикл бенчмарка
# =====================================================================
def run_comprehensive_benchmark():
    scenarios = get_predefined_scenarios()
    os_factor, csv_file = 4, 'benchmark_results.csv'
    file_exists = os.path.isfile(csv_file)

    header_fmt = f"{'Solver':18} | {'Scenario':10} | {'EVM(%)':7} | {'ACLR(dB)':9} | {'Cond(A)':9} | {'Time(ms)':8}"
    print(f"\nЗапуск глобального бенчмарка\n" + "="*len(header_fmt) + f"\n{header_fmt}\n" + "="*len(header_fmt))

    with open(csv_file, mode='a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['Solver', 'Scenario', 'EVM', 'ACLR', 'PAPR', 'Cond_A', 'Time_ms', 'Timestamp'])

        for s_name, cfg in scenarios.items():
            # 1. Сигнал
            x_raw, fs, bw = simulate_ofdm_frame(cfg['n_frames'], cfg['n_fft'], cfg['cp_len'], cfg['n_active'])
            x_os = oversample_fft(x_raw, os_factor)
            x_max, backoff = np.max(np.abs(x_os)), cfg.get('backoff', 0.89)
            x_target_norm = (x_os / x_max) * backoff

            # 2. Reference (без DPD)
            y_no_dpd = power_amplifier_model(x_os, mode='AB', **cfg['pa_params'])
            aclr_no = calculate_aclr(y_no_dpd, fs * os_factor, bw)
            evm_no = calculate_evm(x_raw, align_signals_strictly(x_os, y_no_dpd)[::os_factor])
            print(f"{'No DPD':18} | {s_name:10} | {evm_no:6.2f}% | {aclr_no:9.1f} | {'-':9} | {'-':8}")

            # 3. Обучение DPD
            train_len = min(4096, len(x_target_norm))
            x_target_train = x_target_norm[:train_len]
            A_apply = create_gmp_matrix(x_target_norm, cfg['p_order'], cfg['m_depth'], cfg['cross_depth'])

            for solver_name, solver_func in solvers.items():
                x_dpd_in, w_total, total_time_ms, cond_a = x_os.copy(), None, 0, 0
                alpha = cfg.get('alpha', 0.1)
                fixed_iterations = cfg.get('iterations', 2)

                for it in range(fixed_iterations):
                    y_pa = power_amplifier_model(x_dpd_in, mode='AB', **cfg['pa_params'])
                    y_fb = channel_emulation(y_pa, snr_db=cfg.get('fb_snr', 50))
                    y_sync = align_signals_strictly(x_os, y_fb)

                    y_train = y_sync[:train_len]
                    
                    # Валидация данных
                    if not np.all(np.isfinite(y_train)):
                        continue

                    phase_corr = np.exp(-1j * np.angle(np.sum(y_train * np.conj(x_target_train))))
                    y_train_norm = (y_train * phase_corr) / (np.max(np.abs(y_train)) + 1e-9)

                    A_train = create_gmp_matrix(y_train_norm, cfg['p_order'], cfg['m_depth'], cfg['cross_depth'])
                    
                    if it == 0:
                        cond_a = np.linalg.cond(A_train) if np.all(np.isfinite(A_train)) else float('inf')

                    w_new, step_time_ms = solver_func(A_train, x_target_train, cfg.get('reg', 1e-6))
                    total_time_ms += step_time_ms

                    # Корректная инициализация весов
                    if w_total is None:
                        w_total = w_new
                    else:
                        w_total = alpha * w_new + (1 - alpha) * w_total

                    v_sat = cfg['pa_params']['v_sat'] * 0.98
                    x_dpd_raw = (A_apply @ w_total) * (x_max / backoff)
                    mag = np.abs(x_dpd_raw)
                    x_dpd_in = np.where(mag > v_sat, (x_dpd_raw / mag) * v_sat, x_dpd_raw)

                # 4. Финальное измерение
                y_final = align_signals_strictly(x_os, power_amplifier_model(x_dpd_in, mode='AB', **cfg['pa_params']))
                y_final *= np.exp(-1j * np.angle(np.sum(y_final * np.conj(x_os))))

                evm_dpd = calculate_evm(x_raw, y_final[::os_factor])
                aclr_dpd = calculate_aclr(y_final, fs * os_factor, bw)

                writer.writerow([
                    solver_name, s_name, round(evm_dpd, 4), round(aclr_dpd, 2),
                    round(calculate_papr(x_dpd_in), 2), f"{cond_a:.2e}",
                    round(total_time_ms, 2), datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                ])
                f.flush()

                print(f"{solver_name:18} | {s_name:10} | {evm_dpd:6.2f}% | {aclr_dpd:9.1f} | {cond_a:9.2e} | {total_time_ms:8.2f}")
            print("-" * len(header_fmt))

if __name__ == "__main__":
    np.random.seed(42)
    run_comprehensive_benchmark()