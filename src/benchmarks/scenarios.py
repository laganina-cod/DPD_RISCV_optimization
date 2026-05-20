def get_predefined_scenarios():
    # 1. Narrow: Здесь всё идеально, оставляем как есть
    narrow = {
        'n_frames': 20, 'n_fft': 512, 'cp_len': 40, 'n_active': 312,
        'pa_params': {'v_sat': 2.0, 'p': 4.0, 'gain_db': 12, 'ampm_coeff': 0.03},
        'fs': 7.68e6, 'bw': 5e6,
        'p_order': 6, 'm_depth': 2, 'cross_depth': 0, # Упростили, чтобы убрать шум
        'reg': 1e-3, 'fb_snr': 50, 'iterations': 4,
        'alpha': 0.1, 'backoff': 0.90 
    }

    # 2. Wide: Дожимаем EVM
    wide = {
        'n_frames': 20,          # Увеличили выборку для обучения
        'n_fft': 2048, 'cp_len': 256, 'n_active': 1200,
        'pa_params': {'v_sat': 1.2, 'p': 2.5, 'gain_db': 15, 'ampm_coeff': 0.08},
        'fs': 30.72e6, 'bw': 18e6,
        'p_order': 4, 'm_depth': 3, 'cross_depth': 1,
        'reg': 5e-4,            # Чуть снизили, чтобы модель была точнее
        'fb_snr': 48,           # Улучшили чистоту фидбека (программное усреднение)
        'iterations': 5,        # 5 итераций с адаптивной Alpha
        'alpha': 0.12, # Адаптивный шаг
        'backoff': 0.78         # Чуть выше, чтобы не заходить в "hard saturation"
    }
    # FreqSel: Фазовая точность
    freqsel = {
        'n_frames': 20,          # Чуть больше данных
        'n_fft': 2048, 'cp_len': 256, 'n_active': 672,
        'pa_params': {'v_sat': 1.0, 'p': 2.0, 'gain_db': 15, 'ampm_coeff': 0.06},
        'fs': 307.2e6, 'bw': 100e6,
        'p_order': 4,           # Подняли с 3 до 4 (это даст 2-3% к EVM)
        'm_depth': 3,           # Подняли с 2 до 3
        'cross_depth': 0, 
        'reg': 5e-1,            # Снизили регуляризацию, позволяя модели быть точнее
        'fb_snr': 45,           # Добавили SNR
        'iterations': 4,        # 4 итерации вместо 3
        'alpha': 0.12, 
        'backoff': 0.85
    }

    # LowFB: Даем алгоритму время "доползти" до цели
    lowfb = {
        'n_frames': 20, 'n_fft': 1024, 'cp_len': 128, 'n_active': 336,
        'pa_params': {'v_sat': 1.0, 'p': 2.0, 'gain_db': 15, 'ampm_coeff': 0.06},
        'fs': 153.6e6, 'bw': 50e6,
        'p_order': 4,           # Тоже 4
        'm_depth': 5,           # Больше памяти для узкой полосы
        'cross_depth': 1,       # Кросс-термы помогут "собрать" EVM
        'reg': 2e-3,            # Средняя регуляризация
        'fb_snr': 50,           # Максимально чистый фидбек
        'iterations': 6,        # Больше итераций для узкой полосы
        'alpha': 0.1,
        'backoff': 0.90
    }
    # 4. Realtime: Точность для быстрых сессий
    realtime = {
        'n_frames': 100, 'n_fft': 128, 'cp_len': 9, 'n_active': 72,
        'pa_params': {'v_sat': 1.4, 'p': 3.0, 'gain_db': 12, 'ampm_coeff': 0.03},
        'fs': 7.68e6, 'bw': 5e6,
        'p_order': 2, 'm_depth': 3, 'cross_depth': 0,
        'reg': 1e-5, 'fb_snr': 50, 'iterations': 3,
        'alpha': 0.1, 'backoff': 0.90
    }

    return {
        'narrow': narrow, 'wide': wide, 
        'freqsel': freqsel, 'lowfb': lowfb,
         'realtime': realtime
    }