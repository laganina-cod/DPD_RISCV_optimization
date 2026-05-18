import numpy as np
from src.amplifier.models import rapp_pa, feedback_path, power_amplifier_model

def test_rapp_pa_shape():
    x = np.random.randn(100) + 1j * np.random.randn(100)
    y = rapp_pa(x)
    assert y.shape == x.shape
    # Усиление не должно обнулять сигнал
    assert np.mean(np.abs(y)) > 0

def test_power_amplifier_model_modes():
    x = np.random.randn(200) + 1j * np.random.randn(200)
    y_ab = power_amplifier_model(x, mode='AB')
    y_lin = power_amplifier_model(x, mode='linear')
    # В жёстком режиме должны быть нелинейные искажения, амплитуда может быть другой
    assert y_ab.shape == x.shape
    assert y_lin.shape == x.shape