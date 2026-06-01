import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_and_visualize(file_path, target_scenario='narrow'):
    # Читаем данные
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Ошибка: Файл {file_path} не найден.")
        return

    # 1. Фильтрация по сценарию
    df_scenario = df[df['Scenario'] == target_scenario].copy()
    
    if df_scenario.empty:
        print(f"Предупреждение: Данные для сценария '{target_scenario}' не найдены.")
        return

    # --- НОВЫЙ БЛОК АНАЛИТИКИ: GFLOPS и Точка насыщения ---
    # Расчет количества операций с плавающей точкой (FLOP) для нормальных уравнений
    # FLOP = 2 * N_s * N_f^2 + (2/3) * N_f^3
    df_scenario['FLOP'] = 2 * df_scenario['n_samples'] * (df_scenario['n_features']**2) + (2/3) * (df_scenario['n_features']**3)
    
    # GFLOPS = FLOP / (Время в секундах * 10^9) = FLOP / (Time_ms * 10^6)
    df_scenario['GFLOPS'] = df_scenario['FLOP'] / (df_scenario['Time_ms'] * 1e6)

    # 2. Построение визуализации (Оставлено без изменений по запросу)
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f"Анализ производительности решателей (Сценарий: {target_scenario})", fontsize=14)

    # График 1: Зависимость времени от n_features
    sns.lineplot(
        data=df_scenario, 
        x='n_features', 
        y='Time_ms', 
        hue='Solver', 
        marker='o',
        ax=axes[0]
    )
    axes[0].set_title("Время выполнения vs Размер задачи")
    axes[0].set_xlabel("Количество признаков (n_features)")
    axes[0].set_ylabel("Время (мс)")

    # График 2: Зависимость относительной ошибки от n_features
    sns.lineplot(
        data=df_scenario, 
        x='n_features', 
        y='RelError', 
        hue='Solver', 
        marker='s',
        ax=axes[1]
    )
    axes[1].set_title("Относительная ошибка vs Размер задачи")
    axes[1].set_xlabel("Количество признаков (n_features)")
    axes[1].set_ylabel("Относительная ошибка (RelError)")
    axes[1].set_yscale("log") 

    plt.tight_layout()
    img_name = f'benchmark_plot_{target_scenario}.png'
    plt.savefig(img_name, dpi=300)
    print(f"[*] Графики сохранены в файл: {img_name}\n")

    # 3. Расширенный статистический анализ
    stats = df_scenario.groupby('Solver').agg({
        'Time_ms': ['mean', 'std'],
        'RelError': 'mean',
        'GFLOPS': 'mean'
    }).reset_index()

    stats.columns = ['Solver', 'Time_mu', 'Time_sigma', 'Error_mu', 'GFLOPS_mean']

    # Поиск точки насыщения кэша (максимальная эффективность GFLOPS перед падением)
    # Находим индексы строк с максимальным GFLOPS для каждого солвера
    idx_max_gflops = df_scenario.groupby('Solver')['GFLOPS'].idxmax()
    saturation_data = df_scenario.loc[idx_max_gflops, ['Solver', 'n_features', 'GFLOPS']]
    saturation_data.rename(columns={'n_features': 'Sat_Nf', 'GFLOPS': 'GFLOPS_peak'}, inplace=True)

    # Объединяем метрики
    final_stats = pd.merge(stats, saturation_data, on='Solver')

    print("=== Результаты статистического и аналитического анализа ===")
    # Форматированный вывод в консоль
    print(f"{'Solver':<20} | {'Time_mu':>8} | {'Error_mu':>10} | {'GFLOPS_avg':>10} | {'GFLOPS_peak':>11} | {'Sat_Nf':>6}")
    print("-" * 75)
    for _, row in final_stats.iterrows():
        print(f"{row['Solver']:<20} | {row['Time_mu']:>8.2f} | {row['Error_mu']:>10.2e} | {row['GFLOPS_mean']:>10.2f} | {row['GFLOPS_peak']:>11.2f} | {row['Sat_Nf']:>6.0f}")

    # 4. Генерация кода для LaTeX с новыми метриками эффективности
    print("\n=== Код для LaTeX ===")
    print("\\begin{table}[h!]")
    print("\\centering")
    # Добавили колонки для GFLOPS и точки насыщения (Sat. $N_f$)
    print("\\resizebox{\\textwidth}{!}{")
    print("\\begin{tabular}{|l|c|c|c|c|c|c|}")
    print("\\hline")
    print("\\textbf{Solver} & \\textbf{$\\mu_{Time}$ (ms)} & \\textbf{$\\sigma_{Time}$} & \\textbf{Mean RelError} & \\textbf{Avg GFLOPS} & \\textbf{Peak GFLOPS} & \\textbf{Sat. $N_f$} \\\\ \\hline")
    
    for _, row in final_stats.iterrows():
        solver = row['Solver'].replace('_', '\\_') # Экранирование подчеркиваний для LaTeX
        print(f"{solver:<25} & {row['Time_mu']:>8.2f} & {row['Time_sigma']:>8.2f} & {row['Error_mu']:>10.2e} & {row['GFLOPS_mean']:>10.2f} & {row['GFLOPS_peak']:>11.2f} & {row['Sat_Nf']:>6.0f} \\\\ \\hline")
    
    print("\\end{tabular}")
    print("}")
    print(f"\\caption{{Статистические характеристики и вычислительная эффективность алгоритмов для сценария \\texttt{{{target_scenario}}}}}")
    print("\\label{tab:solver_stats_" + target_scenario + "}")
    print("\\end{table}")

if __name__ == "__main__":
    # Для теста вызываем сценарий realtime (или любой другой из ваших данных)
    analyze_and_visualize('crossover_results.csv', target_scenario='realtime')