import numpy as np
import matplotlib.pyplot as plt

# Функция f(x)
def f(x):
    return 2 * np.sin(x + 0.5) - 1.5 + x

# Первая производная функции f'(x)
def df(x):
    return 2 * np.cos(x + 0.5) + 1

# Вторая производная функции f''(x)
def ddf(x):
    return -2 * np.sin(x + 0.5)

# Комбинированный метод хорд и касательных
def combined_chord_tangent_method(a, b, tol, max_iter=100):
    # Определение количества знаков после запятой для вывода
    precision = int(-np.log10(tol)) + 1
    
    # Начальное приближение по методу хорд
    x_n = b - (f(b) * (a - b)) / (f(a) - f(b))
    iter_count = 0
    
    print(f"{'Шаг':>10}{'an':>20}{'a(n+1)':>20}{'|an-a(n+1)|':>20}{'Точность':>20}"
          f"{'bn':>20}{'b(n+1)':>20}{'|bn-b(n+1)|':>20}{'Точность':>20}")
    
    while True:
        # Применение метода касательных (Ньютона)
        x_n_plus_1 = x_n - f(x_n) / df(x_n)
        
        # Расчет новых границ интервала
        a_new = x_n_plus_1 - (f(x_n_plus_1) * (x_n - x_n_plus_1)) / (f(x_n) - f(x_n_plus_1))
        b_new = x_n_plus_1
        
        # Разница и сравнение с точностью
        da_diff = abs(a_new - a)
        db_diff = abs(b_new - b)
        da_comparison = 'меньше' if da_diff < tol else 'больше'
        db_comparison = 'меньше' if db_diff < tol else 'больше'
        
        # Вывод значений с нужной точностью
        print(f"{iter_count:>10d}{a:>20.{precision}f}{a_new:>20.{precision}f}{da_diff:>20.{precision}f}{da_comparison:>20}"
              f"{b:>20.{precision}f}{b_new:>20.{precision}f}{db_diff:>20.{precision}f}{db_comparison:>20}")
        
        if da_diff < tol and db_diff < tol:
            break
            
        a = a_new
        b = b_new
        x_n = x_n_plus_1
        iter_count += 1
    
    # Финальный вывод с соблюдением точности
    final_root = (a + b) / 2
    print(f"\nРешение уравнения: X = {final_root:.{precision}f} с точностью {tol:.{precision}f}")

# Основная программа
if __name__ == "__main__":
    # Отображение графика функции на интервале от -10 до 10
    x_values = np.linspace(-10, 10, 200)
    y_values = f(x_values)
    
    plt.figure(figsize=(8, 6))
    plt.plot(x_values, y_values, label='f(x)')
    plt.grid(True)
    plt.title('График функции на интервале [-10, 10]')
    plt.legend()
    plt.show()
    
    # Запрос данных у пользователя
    print("\nВведите границы интервала и точность для комбинированного метода хорд и касательных:")
    a = float(input("Левый край интервала: "))
    b = float(input("Правый край интервала: "))
    tol = float(input("Точность (например, 0.0001): "))
    
    # Создание трех графиков: основной функции, первой и второй производной
    x_values_interval = np.linspace(a, b, 200)
    y_values_f = f(x_values_interval)
    y_values_df = df(x_values_interval)
    y_values_ddf = ddf(x_values_interval)
    
    # График функции на интервале [a, b]
    plt.figure(figsize=(8, 6))
    plt.plot(x_values_interval, y_values_f, label='f(x)')
    plt.axhline(y=0, color='k', linestyle='-', label='нулевая линия')  # Нулевая линия
    plt.grid(True)
    plt.title(f'График функции на интервале [{a}, {b}]')
    plt.legend()
    plt.show()
    
    # График первой производной на интервале [a, b]
    plt.figure(figsize=(8, 6))
    plt.plot(x_values_interval, y_values_df, label="f'(x)")
    plt.axhline(y=0, color='k', linestyle='-', label='нулевая линия')  # Нулевая линия
    plt.grid(True)
    plt.title(f'График первой производной на интервале [{a}, {b}]')
    plt.legend()
    plt.show()
    
    # График второй производной на интервале [a, b]
    plt.figure(figsize=(8, 6))
    plt.plot(x_values_interval, y_values_ddf, label="f''(x)")
    plt.axhline(y=0, color='k', linestyle='-', label='нулевая линия')  # Нулевая линия
    plt.grid(True)
    plt.title(f'График второй производной на интервале [{a}, {b}]')
    plt.legend()
    plt.show()
    
    # Решения уравнения методом хорд и касательных
    print("\nРешая уравнение комбинированным методом хорд и касательных...")
    combined_chord_tangent_method(a, b, tol)