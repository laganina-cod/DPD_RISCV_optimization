import numpy as np

def rapp_pa(x, v_sat=1.0, p=2.0, gain_db=15, ampm_coeff=0.08):
    gain_lin = 10**(gain_db / 20)
    x_in = x * gain_lin
    mag = np.abs(x_in)
    am_am = 1.0 / (1.0 + (mag / v_sat)**(2 * p))**(1.0 / (2 * p))
    am_pm = ampm_coeff * (mag / v_sat)**2
    return x_in * am_am * np.exp(1j * am_pm)

def feedback_path(y, delay_samples=4, snr_db=50):
    p_sig = np.mean(np.abs(y)**2)
    p_noise = p_sig / (10**(snr_db / 10))
    noise = np.sqrt(p_noise/2) * (np.random.randn(*y.shape) + 1j*np.random.randn(*y.shape))
    y_noisy = y + noise
    if delay_samples > 0:
        y_noisy = np.roll(y_noisy, delay_samples)
    return y_noisy

# Добавьте эти функции, чтобы убрать ImportError
def power_amplifier_model(x, mode='AB', **kwargs):
    """Обертка для совместимости"""
    if mode == 'AB':
        return rapp_pa(x, **kwargs)
    return x # Линейный режим

def channel_emulation(signal, snr_db):
    """Обертка для совместимости"""
    sig_power = np.mean(np.abs(signal)**2)
    noise_power = sig_power / (10**(snr_db / 10))
    noise = np.sqrt(noise_power/2) * (np.random.randn(len(signal)) + 1j*np.random.randn(len(signal)))
    return signal + noise