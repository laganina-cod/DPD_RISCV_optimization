def run_crossover_benchmark():
    scenarios = get_predefined_scenarios()
    train_len = 4096
    csv_file = 'crossover_results.csv'
    file_exists = os.path.isfile(csv_file)

    # 1. Warm-up (прогрев системы)
    # Это предотвращает искажения данных из-за того, что CPU начинает с низкого энергопотребления
    print("Прогрев системы для стабилизации частоты и кэша...")
    dummy_A = np.random.randn(1024, 64) + 1j * np.random.randn(1024, 64)
    dummy_y = np.random.randn(1024) + 1j * np.random.randn(1024)
    for _ in range(20):
        _ = solvers['Gauss_Winograd'](dummy_A, dummy_y, 1e-6)

    # Сетка параметров: перебор комбинаций для исследования масштабируемости
    param_grid = []
    for p in [3, 5, 7, 9, 11]:
        for m in [2, 3, 4, 5]:
            for c in [0, 1, 2]:
                param_grid.append({'p_order': p, 'm_depth': m, 'cross_depth': c})

    with open(csv_file, mode='a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['Scenario', 'Solver', 'n_features', 'n_samples', 'Time_ms', 'RelError', 'Timestamp'])

        for scen_name, cfg in scenarios.items():
            print(f"\n=== Сценарий: {scen_name} ===")
            y_norm, target, _, _, _, _, _, _, _ = generate_training_data(cfg, train_len)

            for params in param_grid:
                A_train = create_gmp_matrix(y_norm, **params)
                n_samples, n_features = A_train.shape
                
                if n_features < 5: continue  

                reg = cfg.get('reg', 1e-6)
                print(f"Тест: n_features={n_features}")

                for solver_name, solver_func in solvers.items():
                    times = []
                    
                    # 2. Статистический цикл: 50 повторений
                    # Используем медиану для исключения случайных "пиков" (прерываний ОС)
                    for _ in range(50):
                        _, elapsed = solver_func(A_train, target, reg)
                        times.append(elapsed)
                    
                    median_time = np.median(times)
                    
                    # 3. Расчет точности (один раз на данном размере матрицы)
                    w, _ = solver_func(A_train, target, reg)
                    residual = A_train @ w - target
                    rel_err = np.linalg.norm(residual) / np.linalg.norm(target)

                    writer.writerow([
                        scen_name, solver_name, n_features, n_samples,
                        round(median_time, 4), f"{rel_err:.6e}", datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    ])
                    f.flush()
                    print(f"  {solver_name:20} | time={median_time:8.3f} ms | err={rel_err:.2e}")

            print("-" * 80)