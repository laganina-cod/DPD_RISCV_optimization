import numpy as np
from scipy.signal import correlate


def calculate_papr(signal):
    """Расчет отношения пиковой мощности к средней (PAPR)."""
    p_peak = np.max(np.abs(signal)**2)
    p_avg = np.mean(np.abs(signal)**2)
    return 10 * np.log10(p_peak / p_avg) if p_avg > 0 else -np.inf


def align_signals_strictly(ref, meas):
    """Синхронизация только по времени (без коррекции фазы и амплитуды)."""
    c = correlate(ref, meas, method='fft', mode='full')
    shift = np.argmax(np.abs(c)) - (len(meas) - 1)
    return np.roll(meas, shift)


def calculate_evm(ref, meas):
    """Расчет Error Vector Magnitude (EVM) в процентах."""
    ref_norm = ref / np.sqrt(np.mean(np.abs(ref)**2))
    meas_norm = meas / np.sqrt(np.mean(np.abs(meas)**2))
    return np.sqrt(np.mean(np.abs(ref_norm - meas_norm)**2)) * 100


def calculate_aclr(signal, fs, bw, max_len=16384):
    """Расчет коэффициента утечки в соседний канал (ACLR)."""
    if len(signal) > max_len:
        signal = signal[:max_len]
    
    n = len(signal)
    window = np.hanning(n)
    S = np.fft.fft(signal * window) / n
    Pxx = np.abs(S)**2
    f = np.fft.fftfreq(n, 1 / fs)
    
    idx = np.argsort(f)
    f = f[idx]
    Pxx = Pxx[idx]
    
    main = (f > -bw / 2) & (f < bw / 2)
    adj_low = (f > -1.5 * bw) & (f < -0.5 * bw)
    adj_high = (f > 0.5 * bw) & (f < 1.5 * bw)
    
    P_main = np.sum(Pxx[main])
    P_adj = max(np.sum(Pxx[adj_low]), np.sum(Pxx[adj_high]))
    
    if P_main <= 0 or P_adj <= 0:
        return -100.0
        
    return 10 * np.log10(P_adj / P_main)


def get_ccdf(signal):
    """Расчет дополнительной интегральной функции распределения (CCDF)."""
    pwr = np.abs(signal)**2
    papr_val = 10 * np.log10(pwr / np.mean(pwr))
    sorted_papr = np.sort(papr_val)
    y_ccdf = 1.0 - np.arange(len(sorted_papr)) / len(sorted_papr)
    return sorted_papr, y_ccdf