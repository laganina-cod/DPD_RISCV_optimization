import pandas as pd
import numpy as np

def analyze_csv(file_path):
    # Читаем данные. На кафедре ВТ уважаем pandas за скорость и удобство
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Ошибка: Файл {file_path} не найден. Сначала запустите runner.")
        return

    # Группируем по сценарию и считаем среднее и СКО
    stats = df.groupby('Scenario').agg({
        'EVM': ['mean', 'std'],
        'ACLR': ['mean', 'std'],
        'Time_ms': ['mean', 'std']
    }).reset_index()

    # Очищаем заголовки после агрегации
    stats.columns = ['Scenario', 'EVM_mu', 'EVM_sigma', 'ACLR_mu', 'ACLR_sigma', 'Time_mu', 'Time_sigma']

    print("=== Результаты статистического анализа ===")
    print(stats.to_string(index=False))

    # Генерация LaTeX таблицы
    print("\n=== Код для LaTeX ===")
    print("\\begin{table}[h!]")
    print("\\centering")
    print("\\begin{tabular}{|l|c|c|c|c|c|}")
    print("\\hline")
    print("\\textbf{Scenario} & \\textbf{$\\mu_{EVM}$ (\\%)} & \\textbf{$\\sigma_{EVM}$} & \\textbf{$\\mu_{ACLR}$ (дБ)} & \\textbf{$\\sigma_{ACLR}$} & \\textbf{Time (ms)} \\\\ \\hline")
    
    for _, row in stats.iterrows():
        print(f"{row['Scenario']:<10} & {row['EVM_mu']:>6.2f} & {row['EVM_sigma']:>5.2f} & {row['ACLR_mu']:>7.1f} & {row['ACLR_sigma']:>5.2f} & {row['Time_mu']:>7.1f} \\\\ \\hline")
    
    print("\\end{tabular}")
    print("\\caption{Статистические характеристики DPD}")
    print("\\end{table}")

if __name__ == "__main__":
    analyze_csv('benchmark_results.csv')