import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d

# =====================================================================
# Настройка: выберите нужный сценарий
# =====================================================================
SCENARIO = 'narrow'          # Возможные: 'narrow', 'wide', 'freqsel', 'lowfb', 'realtime'
CSV_FILE = 'crossover_results.csv'

# =====================================================================
# Загрузка данных
# =====================================================================
df = pd.read_csv(CSV_FILE)

# Оставляем только строки для выбранного сценария
df_scenario = df[df['Scenario'] == SCENARIO].copy()

if df_scenario.empty:
    print(f"Нет данных для сценария '{SCENARIO}'. Проверьте CSV-файл и название сценария.")
    exit()

# Группируем по solver и n_features, усредняем время (если есть повторы)
df_avg = df_scenario.groupby(['Solver', 'n_features'], as_index=False)['Time_ms'].mean()

# =====================================================================
# Построение графика
# =====================================================================
plt.figure(figsize=(12, 7))
solvers_list = df_avg['Solver'].unique()
colors = plt.cm.tab10(np.linspace(0, 1, len(solvers_list)))

for idx, solver in enumerate(solvers_list):
    subset = df_avg[df_avg['Solver'] == solver].sort_values('n_features')
    plt.plot(subset['n_features'], subset['Time_ms'], 'o-', color=colors[idx], label=solver, markersize=4)

plt.xlabel('Количество признаков (n_features)', fontsize=12)
plt.ylabel('Среднее время решения (мс)', fontsize=12)
plt.title(f'Зависимость времени от размерности матрицы, сценарий: {SCENARIO}', fontsize=14)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)

# Опционально: поиск и отображение точек пересечения (кроссовер-поинтов)
# между всеми парами солверов, используя линейную интерполяцию
find_crossovers = True
if find_crossovers:
    crossover_points = []
    for i in range(len(solvers_list)):
        for j in range(i+1, len(solvers_list)):
            s1 = solvers_list[i]
            s2 = solvers_list[j]
            # Получаем данные двух кривых
            d1 = df_avg[df_avg['Solver'] == s1].dropna().sort_values('n_features')
            d2 = df_avg[df_avg['Solver'] == s2].dropna().sort_values('n_features')
            if d1.empty or d2.empty:
                continue
            # Определяем общий диапазон n_features
            x_min = max(d1['n_features'].min(), d2['n_features'].min())
            x_max = min(d1['n_features'].max(), d2['n_features'].max())
            if x_min >= x_max:
                continue
            # Интерполируем на общую сетку
            x_common = np.unique(np.sort(np.concatenate([d1['n_features'].values, d2['n_features'].values])))
            x_common = x_common[(x_common >= x_min) & (x_common <= x_max)]
            if len(x_common) < 2:
                continue
            f1 = interp1d(d1['n_features'], d1['Time_ms'], kind='linear', bounds_error=False, fill_value='extrapolate')
            f2 = interp1d(d2['n_features'], d2['Time_ms'], kind='linear', bounds_error=False, fill_value='extrapolate')
            y1 = f1(x_common)
            y2 = f2(x_common)
            # Ищем пересечения (смена знака разности)
            diff = y1 - y2
            for k in range(len(x_common)-1):
                if diff[k] == 0:
                    crossover_points.append((x_common[k], y1[k], s1, s2))
                elif diff[k] * diff[k+1] < 0:
                    # Линейная интерполяция точки пересечения
                    t = -diff[k] / (diff[k+1] - diff[k])
                    x_cross = x_common[k] + t * (x_common[k+1] - x_common[k])
                    y_cross = y1[k] + t * (y1[k+1] - y1[k])
                    crossover_points.append((x_cross, y_cross, s1, s2))
    # Отмечаем точки пересечения на графике
    for x_c, y_c, s1, s2 in crossover_points:
        plt.plot(x_c, y_c, 'rx', markersize=8, markeredgewidth=2, markeredgecolor='black')
        # plt.annotate(f'{x_c:.0f}', (x_c, y_c), textcoords="offset points", xytext=(0,10), ha='center', fontsize=7)

plt.tight_layout()
plt.show()