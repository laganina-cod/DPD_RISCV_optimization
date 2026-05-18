import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import welch

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.signals.generation import simulate_ofdm_frame
from src.signals.metrics import calculate_papr, calculate_aclr, calculate_evm
from src.amplifier.models import power_amplifier_model, feedback_path
from src.dpd.base import DigitalPreDistorter

FS = 30.72e6
BW = 18e6

print("--- Запуск эксперимента: Линеаризация в жестком режиме ---")
print("Генерация базового LTE-сигнала...")
tx_base = simulate_ofdm_frame(rb_count=25, m_order=16, seed=123)

# Back-off: пиковая амплитуда 0.1
peak = np.max(np.abs(tx_base))
tx = tx_base / peak * 0.1

# Усилитель
print("Прохождение через усилитель без предыскажений...")
pa_out_no_dpd = power_amplifier_model(tx, mode='AB')
fb = feedback_path(pa_out_no_dpd, delay_samples=4, snr_db=38)

# Обучение DPD (indirect)
print("Обучение DPD (обратная модель)...")
dpd = DigitalPreDistorter(order=5, algo='perminov')
dpd.train(fb, tx)   # выход PA -> цель чистый сигнал

# Применение
print("Применение предыскажений...")
tx_pred = dpd.apply(tx)
pa_out_with_dpd = power_amplifier_model(tx_pred, mode='AB')

# Метрики
evm_no = calculate_evm(tx, pa_out_no_dpd)
evm_dpd = calculate_evm(tx, pa_out_with_dpd)
aclr_no = calculate_aclr(pa_out_no_dpd, FS, BW)
aclr_dpd = calculate_aclr(pa_out_with_dpd, FS, BW)
papr_no = calculate_papr(pa_out_no_dpd)
papr_dpd = calculate_papr(pa_out_with_dpd)

results = {
    "Метрика": ["EVM (%)", "ACLR (дБ)", "PAPR (дБ)"],
    "Без DPD": [round(evm_no, 2), round(aclr_no, 2), round(papr_no, 2)],
    "С DPD":  [round(evm_dpd, 2), round(aclr_dpd, 2), round(papr_dpd, 2)]
}
print("\n--- Итоговые результаты ---")
print(pd.DataFrame(results).to_string(index=False))

# Визуализация спектров
def plot_spectra(original, distorted, linearized, fs):
    plt.figure(figsize=(10, 6))
    for sig, label, color in zip([original, distorted, linearized],
                                 ['Исходный', 'Без DPD', 'С DPD'],
                                 ['black', 'red', 'blue']):
        f, Pxx = welch(sig, fs, nperseg=2048, return_onesided=False)
        Pxx_db = 10 * np.log10(np.fft.fftshift(Pxx) / np.max(Pxx))
        f_mhz = np.fft.fftshift(f) / 1e6
        plt.plot(f_mhz, Pxx_db, label=label, color=color)
    plt.title('Спектральная плотность мощности')
    plt.xlabel('Частота (МГц)')
    plt.ylabel('Относительная мощность (дБ)')
    plt.grid(True, linestyle='--')
    plt.legend()
    plt.ylim([-80, 5])
    plt.xlim([-15, 15])
    plt.tight_layout()
    plt.savefig('dpd_spectrum_comparison.png', dpi=300)
    print("\nГрафик спектра сохранён как 'dpd_spectrum_comparison.png'")
    plt.show()

plot_spectra(tx, pa_out_no_dpd, pa_out_with_dpd, FS)