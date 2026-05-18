def get_predefined_scenarios():
    narrow = {
        'n_frames': 3,
        'n_symbols_per_frame': 72,
        'n_fft': 512,
        'cp_len': 40,
        'n_active': 312,
        'pa_params': {'v_sat': 1.2, 'p': 3.0, 'gain_db': 12, 'ampm_coeff': 0.03},
        'fs': 7.68e6, 'bw': 5e6,
        'p_order': 5, 'm_depth': 2, 'reg': 1e-5,
        'fb_snr': 38
    }
    wide = {
        'n_frames': 2,
        'n_symbols_per_frame': 600,
        'n_fft': 2048,
        'cp_len': 256,
        'n_active': 1200,
        'pa_params': {'v_sat': 0.9, 'p': 2.0, 'gain_db': 15, 'ampm_coeff': 0.08},
        'fs': 30.72e6, 'bw': 18e6,
        'p_order': 9, 'm_depth': 6, 'reg': 1e-4,
        'fb_snr': 38
    }
    agg = {
        'n_frames': 2,
        'n_symbols_per_frame': 1200,
        'n_fft': 4096,
        'cp_len': 512,
        'n_active': 1332,
        'pa_params': {'v_sat': 0.8, 'p': 1.5, 'gain_db': 18, 'ampm_coeff': 0.12},
        'fs': 614.4e6, 'bw': 200e6,
        'p_order': 11, 'm_depth': 5, 'reg': 1e-2,
        'fb_snr': 38
    }
    freqsel = {
        'n_frames': 3,
        'n_symbols_per_frame': 600,
        'n_fft': 2048,
        'cp_len': 256,
        'n_active': 672,
        'pa_params': {'v_sat': 1.0, 'p': 2.0, 'gain_db': 15, 'ampm_coeff': 0.06},
        'fs': 307.2e6, 'bw': 100e6,
        'p_order': 7, 'm_depth': 4, 'reg': 1e-4,
        'fb_snr': 38
    }
    lowfb = {
        'n_frames': 3,
        'n_symbols_per_frame': 600,
        'n_fft': 1024,
        'cp_len': 128,
        'n_active': 336,
        'pa_params': {'v_sat': 1.0, 'p': 2.0, 'gain_db': 15, 'ampm_coeff': 0.06},
        'fs': 153.6e6, 'bw': 50e6,
        'p_order': 7, 'm_depth': 3, 'reg': 1e-4,
        'fb_snr': 25   # намеренно низкий SNR
    }
    strong_mem = {
        'n_frames': 1,
        'n_symbols_per_frame': 2400,
        'n_fft': 8192,
        'cp_len': 1024,
        'n_active': 2664,
        'pa_params': {'v_sat': 0.85, 'p': 1.8, 'gain_db': 18, 'ampm_coeff': 0.12},
        'fs': 1228.8e6, 'bw': 400e6,
        'p_order': 11, 'm_depth': 9, 'reg': 5e-1,
        'fb_snr': 38
    }
    realtime = {
        'n_frames': 50,
        'n_symbols_per_frame': 72,
        'n_fft': 128,
        'cp_len': 9,
        'n_active': 72,
        'pa_params': {'v_sat': 1.2, 'p': 3.0, 'gain_db': 12, 'ampm_coeff': 0.03},
        'fs': 7.68e6, 'bw': 5e6,
        'p_order': 5, 'm_depth': 2, 'reg': 1e-5,
        'fb_snr': 38
    }
    mimo2 = {
        'n_frames': 3,
        'n_symbols_per_frame': 600,
        'n_fft': 2048,
        'cp_len': 144,
        'n_active': 672,
        'pa_params': {'v_sat': 1.0, 'p': 2.0, 'gain_db': 15, 'ampm_coeff': 0.06},
        'fs': 307.2e6, 'bw': 100e6,
        'p_order': 7, 'm_depth': 4, 'reg': 1e-4,
        'n_ant': 2,
        'fb_snr': 38
    }
    return {
        'narrow': narrow,
        'wide': wide,
        'agg': agg,
        'freqsel': freqsel,
        'lowfb': lowfb,
        'strong_mem': strong_mem,
        'realtime': realtime,
        'mimo2': mimo2
    }