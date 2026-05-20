import numpy as np
import pytest
from src.signals.generation import generate_qam_symbols, ofdm_modulate

def quantize_signal(x, bits=12):
    """
    Линейное квантование сигнала до заданного количества бит.
    """
    if bits is None:
        return x
    
    x = np.asarray(x)
    if x.size == 0:
        return x
        
    # Находим максимум для нормировки (симуляция полной шкалы АЦП)
    max_val = np.max(np.abs(x))
    if max_val == 0:
        return x
        
    # Количество уровней (для 12 бит = 2048 уровней на плечо)
    levels = 2**(bits - 1)
    
    # Квантование
    x_norm = x / max_val
    x_q = np.round(x_norm * (levels - 1)) / (levels - 1)
    
    return x_q * max_val

def test_quantization_effect():
    """Проверка работы квантования."""
    x = np.array([0.555555, 0.888888], dtype=complex)
    # Квантуем до 4 бит (очень грубо)
    x_q = quantize_signal(x, bits=4)
    
    assert not np.array_equal(x, x_q)
    assert len(x_q) == len(x)

def test_ofdm_output_type():
    """Проверка, что OFDM генератор возвращает комплексный массив."""
    syms = generate_qam_symbols(600, 16)
    sig = ofdm_modulate(syms, n_fft=1024, cp_len=72, n_active=600)
    assert sig.dtype == complex or sig.dtype == np.complex128