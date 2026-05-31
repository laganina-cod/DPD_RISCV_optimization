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
    df_scenario = df[df['Scenario'] == target_scenario]
    
    if df_scenario.empty:
        print(f"Предупреждение: Данные для сценария '{target_scenario}' не найдены.")
        return

    # 2. Построение визуализации
    # Используем seaborn для автоматической группировки по 'Solver' и расчета доверительных интервалов
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
    # Для ошибки часто полезен логарифмический масштаб, если разброс значений велик
    axes[1].set_yscale("log") 

    plt.tight_layout()
    # Сохраняем график в файл
    img_name = f'benchmark_plot_{target_scenario}.png'
    plt.savefig(img_name, dpi=300)
    print(f"[*] Графики сохранены в файл: {img_name}\n")
    
    # Раскомментируйте строку ниже, если запускаете в Jupyter/интерактивной среде
    # plt.show()

    # 3. Статистический анализ для LaTeX
    # Группируем по решателю для сводной таблицы
    stats = df_scenario.groupby('Solver').agg({
        'Time_ms': ['mean', 'std'],
        'RelError': ['mean']
    }).reset_index()

    stats.columns = ['Solver', 'Time_mu', 'Time_sigma', 'Error_mu']

    print("=== Результаты статистического анализа ===")
    print(stats.to_string(index=False))

    # 4. Генерация кода для LaTeX
    print("\n=== Код для LaTeX ===")
    print("\\begin{table}[h!]")
    print("\\centering")
    print("\\begin{tabular}{|l|c|c|c|}")
    print("\\hline")
    print("\\textbf{Solver} & \\textbf{$\\mu_{Time}$ (ms)} & \\textbf{$\\sigma_{Time}$} & \\textbf{Mean RelError} \\\\ \\hline")
    
    for _, row in stats.iterrows():
        # Ошибку выводим в экспоненциальном формате (e.g., 1.68e-01) для компактности
        print(f"{row['Solver']:<20} & {row['Time_mu']:>8.2f} & {row['Time_sigma']:>8.2f} & {row['Error_mu']:>10.2e} \\\\ \\hline")
    
    print("\\end{tabular}")
    print(f"\\caption{{Статистические характеристики решателей для сценария \\texttt{{{target_scenario}}}}}")
    print("\\label{tab:solver_stats_" + target_scenario + "}")
    print("\\end{table}")

if __name__ == "__main__":
    # Вызываем функцию для конкретного сценария (можно поменять на другой при необходимости)
    analyze_and_visualize('risc_v_results/crossover_results.csv', target_scenario='realtime')