import numpy as np


def channel_emulation(signal, snr_db):
    """
    Эмуляция канала для приемника (добавление AWGN шума).
    """
    # Расчет средней мощности сигнала
    sig_power = np.mean(np.abs(signal)**2)
    
    # Расчет мощности шума исходя из заданного SNR (в дБ)
    noise_power = sig_power / (10**(snr_db / 10))
    
    # Генерация комплексного белого гауссовского шума
    # Стандартное отклонение распределяется между реальной и мнимой частями
    noise_std = np.sqrt(noise_power / 2)
    noise = noise_std * (np.random.randn(len(signal)) + 1j * np.random.randn(len(signal)))
    
    return signal + noise