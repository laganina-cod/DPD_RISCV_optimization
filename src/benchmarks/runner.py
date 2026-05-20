import os
import csv
import time
import numpy as np
from datetime import datetime
from scipy.signal import correlate

from src.benchmarks.scenarios import get_predefined_scenarios
from src.dpd.solvers import qr_householder_solver
from src.amplifier.models import power_amplifier_model
from src.amplifier.feedback_path import channel_emulation
from src.signals.generation import simulate_ofdm_frame
from src.signals.metrics import calculate_evm, calculate_aclr, calculate_papr

def align_signals_strictly(ref, meas):
    """Корреляционная синхронизация сигналов по времени."""
    c = correlate(ref, meas, method='fft', mode='full')
    s = np.argmax(np.abs(c)) - (len(meas) - 1)
    return np.roll(meas, s)

def oversample_fft(signal, os_factor):
    """Оверсемплинг сигнала через дополнение спектра нулями."""
    n = len(signal)
    n_os = n * os_factor
    spec = np.fft.fft(signal)
    half = (n + 1) // 2
    spec_os = np.zeros(n_os, dtype=complex)
    spec_os[:half] = spec[:half]
    spec_os[-(n - half):] = spec[half:]
    return np.fft.ifft(spec_os) * os_factor

def create_gmp_matrix(x, p_order=3, m_depth=2, cross_depth=1):
    """Построение матрицы GMP."""
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

def run_comprehensive_benchmark(solver_name, solver_func):
    """Основной цикл тестирования."""
    scenarios = get_predefined_scenarios()
    os_factor, csv_file = 4, 'benchmark_results.csv'
    file_exists = os.path.isfile(csv_file)

    header_fmt = f"{'Mode':10} | {'Scenario':11} | {'EVM(%)':7} | {'ACLR(dB)':9} | {'Cond(A)':9} | {'Time(ms)':8}"
    print(f"\nЗапуск бенчмарка: {solver_name.upper()}\n" + "="*len(header_fmt) + f"\n{header_fmt}\n" + "="*len(header_fmt))

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

            # 2. Reference
            y_no_dpd = power_amplifier_model(x_os, mode='AB', **cfg['pa_params'])
            aclr_no = calculate_aclr(y_no_dpd, fs * os_factor, bw)
            evm_no = calculate_evm(x_raw, align_signals_strictly(x_os, y_no_dpd)[::os_factor])

            # 3. Обучение DPD
            train_len = min(4096, len(x_target_norm))
            x_target_train = x_target_norm[:train_len]
            A_apply = create_gmp_matrix(x_target_norm, cfg['p_order'], cfg['m_depth'], cfg['cross_depth'])
            
            x_dpd_in, w_total, total_time_ms, cond_a = x_os.copy(), None, 0, 0
            alpha = cfg.get('alpha', 0.1)

            for it in range(cfg.get('iterations', 2)):
                y_pa = power_amplifier_model(x_dpd_in, mode='AB', **cfg['pa_params'])
                y_fb = channel_emulation(y_pa, snr_db=cfg.get('fb_snr', 50))
                y_sync = align_signals_strictly(x_os, y_fb)
                
                y_train = y_sync[:train_len]
                phase_corr = np.exp(-1j * np.angle(np.sum(y_train * np.conj(x_target_train))))
                y_train_norm = (y_train * phase_corr) / np.max(np.abs(y_train))
                
                A_train = create_gmp_matrix(y_train_norm, cfg['p_order'], cfg['m_depth'], cfg['cross_depth'])
                if it == 0: cond_a = np.linalg.cond(A_train)

                start = time.perf_counter()
                # Солвер теперь возвращает только w
                w_new = np.asarray(solver_func(A_train, x_target_train, reg=cfg['reg']))
                total_time_ms += (time.perf_counter() - start) * 1000
                
                w_total = w_new if w_total is None else alpha * w_new + (1 - alpha) * w_total

                # Применение
                v_sat = cfg['pa_params']['v_sat'] * 0.98
                x_dpd_raw = (A_apply @ w_total) * (x_max / backoff)
                mag = np.abs(x_dpd_raw)
                x_dpd_in = np.where(mag > v_sat, (x_dpd_raw / mag) * v_sat, x_dpd_raw)

            # 4. Финал
            y_final = align_signals_strictly(x_os, power_amplifier_model(x_dpd_in, mode='AB', **cfg['pa_params']))
            y_final *= np.exp(-1j * np.angle(np.sum(y_final * np.conj(x_os))))
            
            evm_dpd = calculate_evm(x_raw, y_final[::os_factor])
            aclr_dpd = calculate_aclr(y_final, fs * os_factor, bw)
            
            writer.writerow([solver_name, s_name, round(evm_dpd, 4), round(aclr_dpd, 2), 
                             round(calculate_papr(x_dpd_in), 2), f"{cond_a:.2e}", 
                             round(total_time_ms, 2), datetime.now().strftime("%Y-%m-%d %H:%M:%S")])
            f.flush()

            print(f"{'No DPD':10} | {s_name:11} | {evm_no:6.2f}% | {aclr_no:9.1f} | {'-':9} | {'-':8}")
            print(f"{'With DPD':10} | {s_name:11} | {evm_dpd:6.2f}% | {aclr_dpd:9.1f} | {cond_a:9.2e} | {total_time_ms:8.2f}")
            print("-" * len(header_fmt))

if __name__ == "__main__":
    np.random.seed(42)
    run_comprehensive_benchmark('Perminov_HPC_Edition', qr_householder_solver)