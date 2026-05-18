import sys
import numpy as np
import time
from scipy.signal import correlate
from src.benchmarks.scenarios import get_predefined_scenarios
from src.dpd.solvers import (
    solver_naive, solver_perminov, solver_strassen, solver_winograd,
    solver_cholesky_winograd, solver_cholesky_perminov, solver_cholesky_strassen,
    solver_cholesky_naive, qr_householder_solver
)
from src.amplifier.models import power_amplifier_model, channel_emulation
from src.signals.generation import simulate_ofdm_frame

# --------------------------------------------------------------
# Вспомогательные функции
# --------------------------------------------------------------
def oversample_fft(signal, os_factor):
    n = len(signal)
    n_os = n * os_factor
    X = np.fft.fft(signal)
    half = n // 2
    X_os = np.zeros(n_os, dtype=complex)
    X_os[:half] = X[:half]
    X_os[-half:] = X[half:]
    return np.fft.ifft(X_os) * os_factor

def align_signals(ref, meas):
    c = correlate(ref, meas, method='fft', mode='full')
    shift = np.argmax(np.abs(c)) - (len(meas) - 1)
    meas_shifted = np.roll(meas, shift)
    ph = np.dot(ref, meas_shifted.conj()) / (np.dot(meas_shifted, meas_shifted.conj()) + 1e-12)
    return meas_shifted * ph

def create_gmp_matrix(x, p_order=5, m_depth=2):
    n = len(x)
    powers = range(1, p_order + 1, 2)
    ncols = m_depth * len(powers)
    A = np.empty((n, ncols), dtype=np.complex128)
    col_idx = 0
    for m in range(m_depth):
        x_del = np.roll(x, m)
        if m > 0:
            x_del[:m] = 0.0
        abs_x_del = np.abs(x_del).astype(np.complex128)
        for p in powers:
            A[:, col_idx] = x_del * (abs_x_del ** (p - 1))
            col_idx += 1
    return A

def filter_low_bandwidth(signal):
    window_size = 3
    return np.convolve(signal, np.ones(window_size) / window_size, mode='same')

# --------------------------------------------------------------
# Безопасные метрики (без scipy.signal.welch, без зависаний)
# --------------------------------------------------------------
def safe_papr(signal):
    p_peak = np.max(np.abs(signal)**2)
    p_avg = np.mean(np.abs(signal)**2)
    return 10 * np.log10(p_peak / p_avg) if p_avg > 0 else -np.inf

def safe_evm(ref, meas, max_len=4096):
    if len(ref) > max_len:
        ref = ref[:max_len]
        meas = meas[:max_len]
    # Амплитудная нормализация (сигналы уже синхронизированы)
    norm_ref = np.linalg.norm(ref)
    norm_meas = np.linalg.norm(meas)
    if norm_meas > 0:
        meas = meas * (norm_ref / norm_meas)
    error = ref - meas
    return (np.linalg.norm(error) / norm_ref) * 100

def safe_aclr(signal, fs, bw, max_len=16384):
    if len(signal) > max_len:
        signal = signal[:max_len]
    n = len(signal)
    window = np.hanning(n)
    S = np.fft.fft(signal * window) / n
    Pxx = np.abs(S)**2
    f = np.fft.fftfreq(n, 1/fs)
    idx = np.argsort(f)
    f = f[idx]
    Pxx = Pxx[idx]
    main = (f > -bw/2) & (f < bw/2)
    adj_low = (f > -1.5*bw) & (f < -0.5*bw)
    adj_high = (f > 0.5*bw) & (f < 1.5*bw)
    P_main = np.sum(Pxx[main])
    P_adj = max(np.sum(Pxx[adj_low]), np.sum(Pxx[adj_high]))
    if P_main <= 0 or P_adj <= 0:
        return -100.0
    return 10 * np.log10(P_adj / P_main)

# --------------------------------------------------------------
# Основной бенчмарк (без отладочных принтов и разогрева)
# --------------------------------------------------------------
def run_comprehensive_benchmark(solver_name, solver_func):
    scenarios = get_predefined_scenarios()
    header = f"{'Mode':10} | {'Scenario':11} | {'EVM(%)':7} | {'ACLR(dB)':9} | {'PAPR(dB)':8} | {'Time(ms)':8}"
    print(f"\nКонфигурация: {solver_name.upper()}")
    print("=" * len(header))
    print(header)
    print("=" * len(header))

    for s_name, cfg in scenarios.items():
        run_seed_offset = 100 if s_name == 'mimo2' else 0

        # Генерация сигнала
        rb_count = cfg['n_active'] // 12
        x_clean = np.concatenate([
            simulate_ofdm_frame(rb_count, n_fft=cfg['n_fft'],
                                cp_len=cfg['cp_len'], seed=i + run_seed_offset)
            for i in range(cfg['n_frames'])
        ])
        x_clean /= np.sqrt(np.mean(np.abs(x_clean)**2))

        # Масштабирование
        v_sat_val = cfg['pa_params']['v_sat']
        # Пик входного сигнала: 90% от входного напряжения насыщения
        target_peak = v_sat_val * 0.3
        x_scaled = x_clean * (target_peak / np.max(np.abs(x_clean)))
        # Оверсемплинг
        os_factor = 4
        x_scaled_os = oversample_fft(x_scaled, os_factor)

        # PA + шум обратной связи
        y_pa_os = power_amplifier_model(x_scaled_os, mode='AB', **cfg['pa_params'])
        y_pa_noisy_os = channel_emulation(y_pa_os, cfg['fb_snr'])
        y_pa_sync_os = align_signals(x_scaled_os, y_pa_noisy_os)

        if s_name == 'lowfb':
            y_pa_sync_os = filter_low_bandwidth(y_pa_sync_os)

        # Нормализация для ILA
        x_max = np.max(np.abs(x_scaled_os))
        y_max = np.max(np.abs(y_pa_sync_os))
        x_train_norm = x_scaled_os / x_max
        y_train_norm = y_pa_sync_os / y_max

        # Матрица GMP
        A_train = create_gmp_matrix(y_train_norm, cfg['p_order'], cfg['m_depth'])

        # Решение СЛАУ (замер времени, без разогрева)
        try:
            t_start = time.perf_counter()
            solver_result = solver_func(A_train, x_train_norm, reg=cfg['reg'])
            t_end = time.perf_counter()
            mean_time_ms = (t_end - t_start) * 1000
            w = solver_result[0] if isinstance(solver_result, tuple) else solver_result
        except Exception as e:
            print(f"Ошибка в солвере {solver_name}: {e}")
            w = np.zeros(A_train.shape[1])
            mean_time_ms = 0.0

        # Применение DPD
        x_verify_norm = x_scaled_os / x_max
        A_apply = create_gmp_matrix(x_verify_norm, cfg['p_order'], cfg['m_depth'])
        x_dpd_norm = A_apply @ w
        x_dpd_os = x_dpd_norm * x_max

        # Защита от пиков
        v_sat_input = v_sat_val
        max_allowed_peak = v_sat_input * 1.3
        mag = np.abs(x_dpd_os)
        over = mag > max_allowed_peak
        if np.any(over):
            x_dpd_os[over] = (x_dpd_os[over] / mag[over]) * max_allowed_peak

        y_final_os = power_amplifier_model(x_dpd_os, mode='AB', **cfg['pa_params'])

        # Шум измерительных приборов
        verify_snr = cfg['fb_snr'] if s_name == 'lowfb' else 55
        y_pa_meas_os = channel_emulation(y_pa_os, verify_snr)
        y_final_meas_os = channel_emulation(y_final_os, verify_snr)

        y_pa_meas_os = align_signals(x_scaled_os, y_pa_meas_os)
        y_final_meas_os = align_signals(x_scaled_os, y_final_meas_os)

        # Децимация (просто прореживание)
        y_pa_ds = y_pa_meas_os[::os_factor]
        y_final_ds = y_final_meas_os[::os_factor]

        # Метрики
        evm_no = safe_evm(x_scaled, y_pa_ds)
        aclr_no = safe_aclr(y_pa_meas_os, cfg['fs'] * os_factor, cfg['bw'])
        papr_no = safe_papr(y_pa_meas_os)

        evm_dpd = safe_evm(x_scaled, y_final_ds)
        aclr_dpd = safe_aclr(y_final_meas_os, cfg['fs'] * os_factor, cfg['bw'])
        papr_dpd = safe_papr(y_final_meas_os)
        print(f"Макс. вход: {np.max(np.abs(x_scaled_os)):.3f}, "
        f"макс. выход: {np.max(np.abs(y_pa_os)):.3f}, "
        f"v_sat: {v_sat_val:.3f}")
        print(f"{'No DPD':10} | {s_name:11} | {evm_no:6.2f}% | {aclr_no:9.1f} | {papr_no:8.2f} | {'-':8}")
        print(f"{'With DPD':10} | {s_name:11} | {evm_dpd:6.2f}% | {aclr_dpd:9.1f} | {papr_dpd:8.2f} | {mean_time_ms:8.2f}")
        print("-" * len(header))

# --------------------------------------------------------------
# Запуск
# --------------------------------------------------------------
if __name__ == "__main__":
    np.random.seed(42)

    all_solvers = {
        'Standart + Perminov Matmul': solver_perminov,
        'Standart + Naive Matmul': solver_naive,
        
        'Standart + Strassen Matmul': solver_strassen,
        'Standart + Winograd Matmul': solver_winograd,
        'Cholesky + Naive Matmul': solver_cholesky_naive,
        'Cholesky + Perminov Matmul': solver_cholesky_perminov,
        'Cholesky + Strassen Matmul': solver_cholesky_strassen,
        'Cholesky + Winograd Matmul': solver_cholesky_winograd,
        'Pure QR Householder': qr_householder_solver
    }

    for name, func in all_solvers.items():
        run_comprehensive_benchmark(name, func)