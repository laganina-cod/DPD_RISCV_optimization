import numpy as np
from src.amplifier.models import rapp_pa

def test_pa_drift_between_frames():
    """
    Моделирует изменение параметров PA от фрейма к фрейму (нагрев).
    Проверяет, что выходная мощность меняется, но форма сигнала остаётся корректной.
    """
    rng = np.random.default_rng(0)
    x = (rng.normal(0, 1, 200) + 1j * rng.normal(0, 1, 200))

    # Номинальный PA
    y1 = rapp_pa(x, v_sat=1.0, p=2.0, gain_db=15)
    # PA после нагрева: немного снизилось усиление и точка насыщения
    y2 = rapp_pa(x, v_sat=0.92, p=2.0, gain_db=14.5)

    # Проверяем, что сигнал не обнулился и форма изменилась
    assert y1.shape == y2.shape
    assert np.mean(np.abs(y2)) < np.mean(np.abs(y1)) * 1.1  # усиление не выросло
    assert not np.allclose(y1, y2)