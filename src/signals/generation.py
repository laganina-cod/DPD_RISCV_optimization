import numpy as np


def generate_qam_symbols(n_symbols, m_order=16, seed=None):
    rng = np.random.default_rng(seed)
    
    # Вычисляем ближайшую сетку (например, для 16-QAM это 4x4)
    sqrt_m = int(np.round(np.sqrt(m_order))) 
    points = np.arange(-(sqrt_m - 1), sqrt_m, 2)
    x, y = np.meshgrid(points, points)
    constellation = x.flatten() + 1j * y.flatten()
    
    # Нормализация мощности
    constellation /= np.sqrt(np.mean(np.abs(constellation)**2))
    
    # КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ: 
    # Используем длину реально созданного созвездия, а не m_order
    actual_m = len(constellation)
    indices = rng.integers(0, actual_m, n_symbols)
    
    return constellation[indices]


def ofdm_modulate(symbols, n_fft=1024, cp_len=72, n_active=600, os_factor=4):
    """
    OFDM-модуляция одного символа с поддержкой оверсемплинга (os_factor).
    Защищает от алиасинга при моделировании нелинейностей УМ и DPD.
    """
    assert n_active <= n_fft, f"n_active={n_active} превышает n_fft={n_fft}"
    assert n_active % 2 == 0, "Число активных поднесущих должно быть чётным"

    if len(symbols) < n_active:
        symbols = np.concatenate([symbols, np.zeros(n_active - len(symbols), dtype=complex)])
    else:
        symbols = symbols[:n_active]

    # Виртуально увеличиваем сетку частот за счет оверсемплинга
    n_fft_os = n_fft * os_factor
    cp_len_os = cp_len * os_factor

    half_act = n_active // 2
    frame_fft = np.zeros(n_fft_os, dtype=complex)
    
    # Распределяем поднесущие вокруг нулевой частоты в увеличенном спектре
    frame_fft[1:half_act + 1] = symbols[:half_act]
    frame_fft[-half_act:] = symbols[half_act:2 * half_act]

    # Переходим во временную область (сигнал теперь дискретизован на частоте fs * os_factor)
    time_data = np.fft.ifft(frame_fft) * np.sqrt(n_fft_os)
    cp = time_data[-cp_len_os:]
    
    return np.concatenate([cp, time_data])


def simulate_ofdm_frame(n_frames=1, n_fft=1024, cp_len=72, n_active=600, m_order=16, os_factor=4, seed=None):
    # Генерируем символы для всех фреймов сразу
    total_symbols = n_active * n_frames
    symbols = generate_qam_symbols(total_symbols, m_order, seed=seed)
    
    all_frames = []
    for i in range(n_frames):
        frame_symbols = symbols[i * n_active:(i + 1) * n_active]
        modulated = ofdm_modulate(frame_symbols, n_fft, cp_len, n_active, os_factor)
        all_frames.append(modulated)
    
    signal = np.concatenate(all_frames)
    
    # Параметры для расчета ACLR в ранере
    fs = 30.72e6  # Дефолтная частота (можно вынести в cfg)
    bw = (n_active / n_fft) * fs
    
    return signal, fs, bw