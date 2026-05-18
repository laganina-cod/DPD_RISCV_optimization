import numpy as np
from scipy.signal import welch, correlate

def сalculate_papr(signal):
    p_peak = np.max(np.abs(signal)**2)
    p_avg = np.mean(np.abs(signal)**2)
    return 10 * np.log10(p_peak / p_avg) if p_avg > 0 else -np.inf

def calculate_evm(ref, meas, max_len=4096):
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

def calculate_aclr(signal, fs, bw, max_len=16384):
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

def get_ccdf(signal):
    pwr = np.abs(signal) ** 2
    papr_val = 10 * np.log10(pwr / np.mean(pwr))
    sorted_papr = np.sort(papr_val)
    y_ccdf = 1.0 - np.arange(len(sorted_papr)) / len(sorted_papr)
    return sorted_papr, y_ccdf
