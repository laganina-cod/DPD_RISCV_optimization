import numpy as np


def power_amplifier_model(x, mode='AB', v_sat=1.0, p=2.0, gain_db=10.0, ampm_coeff=0.05, **kwargs):
    x = np.asarray(x)
    # 1. Защита от пустого входа (исправляет ValueError в np.convolve)
    if x.size == 0:
        return x
    
    gain_lin = 10**(gain_db / 20)
    x_in = x * gain_lin

    # 2. Эмуляция памяти (FIR-фильтр)
    mem_filter = np.array([1.0, 0.15 + 0.05j, 0.05])
    
    # Чтобы длина выхода была РАВНА длине входа, используем свертку 
    # и берем только первые len(x) элементов (причинно-следственная связь)
    x_mem = np.convolve(x_in, mem_filter, mode='full')[:len(x)]

    if mode == 'AB':
        mag = np.abs(x_mem)
        # AM-AM (Rapp)
        am_am = 1.0 / (1.0 + (mag / v_sat)**(2 * p))**(1.0 / (2 * p))
        # AM-PM
        am_pm = ampm_coeff * (mag / v_sat)**2
        return x_mem * am_am * np.exp(1j * am_pm)
    
    # Линейный режим (теперь тоже с учетом фильтра памяти и корректной длины)
    return x_mem