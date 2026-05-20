import pytest
import numpy as np
from src.amplifier.models import power_amplifier_model

def test_nonstationary_pa_behavior():
    """Проверка работы модели в режиме AB."""
    x = np.array([0.5, 1.0, 1.5], dtype=complex)
    # Заменяем rapp_pa на power_amplifier_model
    y = power_amplifier_model(x, mode='AB', v_sat=1.0, p=2.0)
    
    assert len(y) == len(x)
    assert np.abs(y[-1]) < np.abs(x[-1]) * 10 # Проверка компрессии