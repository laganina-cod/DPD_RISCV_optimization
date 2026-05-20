import numpy as np
import pytest
from src.amplifier.models import power_amplifier_model

def quantize_signal(x, bits=12):

    """Обязательно добавьте эту функцию обратно в models.py"""

    if bits is None: return x

    x = np.asarray(x)

    if x.size == 0: return x

   

    max_val = np.max(np.abs(x))

    if max_val == 0: return x

   

    levels = 2**(bits - 1)

    x_norm = x / max_val

    x_q = np.round(x_norm * (levels - 1)) / (levels - 1)

    return x_q * max_val

def test_pa_gain():
    """Проверка линейного усиления (gain_db)."""
    x = np.array([0.1 + 0.1j])
    gain_db = 20.0  # Усиление в 10 раз по напряжению
    # В линейном режиме y[0] должно быть x[0] * 10
    y = power_amplifier_model(x, mode='Linear', gain_db=gain_db)
    
    np.testing.assert_allclose(np.abs(y), np.abs(x) * 10.0, rtol=1e-5)

def test_pa_saturation_ab():
    """Проверка насыщения в режиме AB (модель Раппа)."""
    v_sat = 1.0
    # Подаем очень большой сигнал, который должен упереться в v_sat
    x_large = np.array([100.0]) 
    y = power_amplifier_model(x_large, mode='AB', v_sat=v_sat, gain_db=0.0)
    
    # В модели Раппа при огромном входе амплитуда стремится к v_sat
    assert np.abs(y[0]) <= v_sat
    assert np.abs(y[0]) > 0.95 

def test_pa_complex_handling():
    """Проверка типов данных и сохранения размерности (с учетом памяти)."""
    x = np.array([0.5 + 0.5j, -0.5 - 0.5j])
    y = power_amplifier_model(x, mode='AB')
    
    assert np.iscomplexobj(y)
    assert len(y) == len(x)

def test_pa_phase_symmetry():
    """
    Проверка фазовой симметрии. 
    При инверсии всего входного вектора, выход должен инвертироваться зеркально.
    """
    x = np.array([0.4 + 0.3j, 0.1 - 0.1j])
    y1 = power_amplifier_model(x, mode='AB')
    y2 = power_amplifier_model(-x, mode='AB')
    
    # Амплитуды должны совпадать (память инвертируется вместе с сигналом)
    np.testing.assert_allclose(np.abs(y1), np.abs(y2), rtol=1e-7)
    # Сами значения должны быть противоположны (y1 == -y2)
    np.testing.assert_allclose(y1, -y2, rtol=1e-7)

def test_pa_memory_effect():
    """
    Проверка наличия памяти. 
    Отклик на импульс [1, 0, 0] должен иметь 'хвосты' на 2-м и 3-м отсчетах.
    """
    x = np.array([1.0, 0.0, 0.0], dtype=complex)
    y = power_amplifier_model(x, mode='Linear', gain_db=0.0)
    
    # Из-за фильтра [1.0, 0.15+0.05j, 0.05] в моделях
    assert np.abs(y[1]) > 0, "Отсутствует эффект памяти (такт 1)"
    assert np.abs(y[2]) > 0, "Отсутствует эффект памяти (такт 2)"
    assert np.isclose(y[1], 0.15 + 0.05j)

def test_quantization_levels():
    """Проверка корректности работы квантования."""
    x = np.linspace(-1, 1, 1000)
    # 2 бита: уровни -1, 0, 1. Ошибка будет большой.
    y_q2 = quantize_signal(x, bits=2)
    # 12 бит: уровни очень плотные. Ошибка будет минимальной.
    y_q12 = quantize_signal(x, bits=12)
    
    error_2 = np.mean(np.abs(x - y_q2))
    error_12 = np.mean(np.abs(x - y_q12))
    
    assert error_2 > error_12
    # Проверяем, что макс. амплитуда сохранилась
    assert np.isclose(np.max(np.abs(y_q12)), np.max(np.abs(x)))

def test_pa_empty_input():
    """Проверка на пустой вход (не должно быть ValueError из-за свертки)."""
    x = np.array([])
    y = power_amplifier_model(x)
    y_q = quantize_signal(x)
    
    assert y.size == 0
    assert y_q.size == 0