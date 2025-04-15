import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def calculate_coefficients(x_values, y_values):
    n = len(x_values)
    x = np.array(x_values)
    y = np.array(y_values)
    
    # 1. Вычисление коэффициентов
    sum_x = np.sum(x)
    sum_y = np.sum(y)
    sum_x2 = np.sum(x**2)
    sum_x3 = np.sum(x**3)
    sum_x4 = np.sum(x**4)
    sum_xy = np.sum(x*y)
    sum_x2y = np.sum(x**2 * y)
    
    A = np.array([
        [sum_x4, sum_x3, sum_x2],
        [sum_x3, sum_x2, sum_x],
        [sum_x2, sum_x, n]
    ])
    B = np.array([sum_x2y, sum_xy, sum_y])
    
    try:
        a, b, c = np.linalg.solve(A, B)
    except np.linalg.LinAlgError:
        a, b, c = np.linalg.lstsq(A, B, rcond=None)[0]
    
    # 2. Расчёт метрик
    y_pred = a*x**2 + b*x + c
    residuals = y - y_pred
    sse = np.sum(residuals**2)
    mse = sse/n
    rmse = np.sqrt(mse)
    r2 = 1 - (sse/np.sum((y - np.mean(y))**2))
    mae = np.mean(np.abs(residuals))
    mape = np.mean(np.abs(residuals/y)) * 100
    shapiro_test = stats.shapiro(residuals)
    
    # 3. Формирование выводов
    conclusions = []
    
    # Оценка R²
    if r2 >= 0.95:
        conclusions.append("✓ Отличная аппроксимация (R² ≥ 0.95)")
    elif r2 >= 0.8:
        conclusions.append("✓ Хорошая аппроксимация (0.8 ≤ R² < 0.95)")
    elif r2 >= 0.5:
        conclusions.append("✓ Удовлетворительная аппроксимация (0.5 ≤ R² < 0.8)")
    else:
        conclusions.append("✗ Плохая аппроксимация (R² < 0.5)")
    
    # Проверка остатков
    if shapiro_test[1] > 0.05:
        conclusions.append("✓ Остатки нормально распределены (p-value > 0.05)")
    else:
        conclusions.append("✗ Остатки не нормальны (p-value ≤ 0.05)")
    
    # Оценка ошибок
    if mape < 10:
        conclusions.append("✓ Точность высокая (MAPE < 10%)")
    elif mape < 20:
        conclusions.append("✓ Точность средняя (10% ≤ MAPE < 20%)")
    else:
        conclusions.append("✗ Точность низкая (MAPE ≥ 20%)")
    
    # 4. Визуализация с выводами
    plt.figure(figsize=(18, 12))
    
    # Главный график
    plt.subplot(2, 2, 1)
    x_curve = np.linspace(min(x), max(x), 100)
    y_curve = a*x_curve**2 + b*x_curve + c
    plt.scatter(x, y, color='red', label='Данные')
    plt.plot(x_curve, y_curve, 'b-', label=f'y = {a:.4f}x² + {b:.4f}x + {c:.4f}')
    plt.title(f'Аппроксимация (R² = {r2:.4f})', pad=20)
    
    # Добавляем выводы на график
    conclusion_text = "\n".join(conclusions)
    plt.text(0.05, 0.95, conclusion_text, 
             transform=plt.gca().transAxes,
             bbox=dict(facecolor='white', alpha=0.8),
             verticalalignment='top')
    
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True)
    
    # График остатков
    plt.subplot(2, 2, 2)
    plt.scatter(x, residuals, color='green', label='Остатки')
    plt.axhline(y=0, color='r', linestyle='--')
    
    # Анализ остатков
    res_conclusions = []
    if np.max(np.abs(residuals)) > 2*rmse:
        res_conclusions.append("• Есть потенциальные выбросы")
    if (residuals > 0).sum() != (residuals < 0).sum():
        res_conclusions.append("• Асимметрия остатков")
    
    if not res_conclusions:
        res_conclusions.append("• Остатки выглядят случайными")
    
    plt.title('Анализ остатков', pad=20)
    plt.text(0.05, 0.95, "\n".join(res_conclusions),
             transform=plt.gca().transAxes,
             bbox=dict(facecolor='white', alpha=0.8),
             verticalalignment='top')
    plt.xlabel('x')
    plt.ylabel('Остатки')
    plt.legend()
    plt.grid(True)
    
    # Гистограмма остатков
    plt.subplot(2, 2, 3)
    plt.hist(residuals, bins=15, color='orange', edgecolor='black', density=True)
    
    # Наложим нормальное распределение
    mu, std = np.mean(residuals), np.std(residuals)
    xmin, xmax = plt.xlim()
    x_norm = np.linspace(xmin, xmax, 100)
    p_norm = stats.norm.pdf(x_norm, mu, std)
    plt.plot(x_norm, p_norm, 'k--', label='Нормальное распределение')
    
    plt.title('Распределение остатков', pad=20)
    plt.xlabel('Остатки')
    plt.ylabel('Плотность')
    plt.legend()
    plt.grid(True)
    
    # Q-Q plot
    plt.subplot(2, 2, 4)
    stats.probplot(residuals, plot=plt)
    plt.title('Q-Q plot остатков', pad=20)
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # 5. Итоговый отчёт
    print("\n" + "="*60)
    print("ИТОГОВЫЙ ОТЧЁТ О КАЧЕСТВЕ АППРОКСИМАЦИИ".center(60))
    print("="*60)
    print(f"\nПолиномиальная модель: y = {a:.6f}x² + {b:.6f}x + {c:.6f}\n")
    
    print("Метрики качества:")
    print(f"- Коэффициент детерминации R²: {r2:.6f}")
    print(f"- Сумма квадратов ошибок (SSE): {sse:.6f}")
    print(f"- Среднеквадратичная ошибка (RMSE): {rmse:.6f}")
    print(f"- Средняя абсолютная ошибка (MAE): {mae:.6f}")
    print(f"- Средняя абсолютная процентная ошибка (MAPE): {mape:.2f}%")
    print(f"- Тест Шапиро-Уилка (нормальность): W = {shapiro_test[0]:.4f}, p-value = {shapiro_test[1]:.4f}")
    
    print("\nВыводы:")
    for conclusion in conclusions:
        print(f"  {conclusion}")
    
    if shapiro_test[1] <= 0.05:
        print("\nРекомендации:")
        print("- Рассмотрите преобразование данных (логарифмирование и др.)")
        print("- Проверьте данные на наличие выбросов")
        print("- Попробуйте робастные методы регрессии")
    
    return a, b, c, {
        'r2': r2,
        'sse': sse,
        'rmse': rmse,
        'mae': mae,
        'mape': mape,
        'shapiro_test': shapiro_test,
        'conclusions': conclusions
    }

# Пример использования
xk = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1,
      1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2]
yk = [-1.89, -2.07, -2.30, -2.26, -2.34, -2.66, -2.88, -2.85, -3.16, -3.49,
      -3.88, -4.22, -4.45, -4.99, -5.36, -5.71, -6.51, -6.76, -7.35, -8.02]

a, b, c, report = calculate_coefficients(xk, yk)