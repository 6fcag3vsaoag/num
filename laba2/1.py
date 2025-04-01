import numpy as np
import matplotlib.pyplot as plt

# Функция f(x)
def f(x):
    return 2 * np.sin(x + 0.5) - 1.5 + x

# Производная функции f'(x)
def df(x):
    return 2 * np.cos(x + 0.5) + 1

# Проверка условия сходимости метода простой итерации
def check_convergence_condition(a, b):
    df_abs_max = np.max(np.abs(df(np.linspace(a, b, 500))))
    if df_abs_max >= 1:
        print(f"ПРЕДУПРЕЖДЕНИЕ: Условие сходимости не выполнено на интервале [{a}, {b}], метод может расходиться.")
        return False
    else:
        print(f"Условие сходимости выполнено на интервале [{a}, {b}].")
        return True

# Метод простой итерации
def simple_iteration(a, b, tol, max_iter=10000):
    # График функции на широком диапазоне
    x_values = np.linspace(-10, 10, 200)
    y_values = f(x_values)
    
    plt.figure(figsize=(8, 6))
    plt.plot(x_values, y_values, label='f(x)')
    plt.grid(True)
    plt.title('График функции')
    plt.legend()
    plt.show()
    
    # Запрашиваем начальные значения и точность
    print("Введите интервал и точность:")
    a = float(input("Начало интервала: "))
    b = float(input("Конец интервала: "))
    tol = float(input("Точность (например, 0.0001): "))
    
    # Проверяем условие сходимости
    convergence_check = check_convergence_condition(a, b)
    
    # Графики функции и производной на интервале [a, b]
    x_values_interval = np.linspace(a, b, 200)
    y_values_f = f(x_values_interval)
    y_values_df = df(x_values_interval)
    
    plt.figure(figsize=(8, 6))
    plt.plot(x_values_interval, y_values_f, label='f(x)')
    plt.grid(True)
    plt.title(f'График функции на интервале [{a}, {b}]')
    plt.legend()
    plt.show()
    
    plt.figure(figsize=(8, 6))
    plt.plot(x_values_interval, y_values_df, label="f'(x)")
    plt.axhline(y=1, color='r', linestyle='--', label='y=1')
    plt.axhline(y=-1, color='r', linestyle='--', label='y=-1')
    plt.grid(True)
    plt.title(f'График производной на интервале [{a}, {b}]')
    plt.legend()
    plt.show()
    
    # Алгоритм простой итерации
    iter_count = 0
    x_n = (a + b) / 2  # Начальное приближение
    x_n_plus_1 = None
    epsilon = int(-np.log10(tol)) + 1  # Количество знаков после запятой для вывода
    
    print(f"{'Шаг':>10}{'Xn':>20}{'X(n+1)':>20}{'|Xn - X(n+1)|':>20}{'Сравнение с точностью':>20}")
    while True:
        x_n_plus_1 = x_n - f(x_n) / df(x_n)  # Формула Ньютона-Рафсона
        
        diff = abs(x_n - x_n_plus_1)
        comparison = 'меньше' if diff < tol else 'больше'
        
        print(f"{iter_count:>10d}{x_n:>20.{epsilon}f}{x_n_plus_1:>20.{epsilon}f}{diff:>20.{epsilon}f}{comparison:>20}")
        
        if diff < tol or iter_count >= max_iter:
            break
            
        x_n = x_n_plus_1
        iter_count += 1
    
    print(f"\nРешение уравнения: X = {x_n_plus_1:.{epsilon}f} с точностью {tol:.{epsilon -1}f}")

simple_iteration(-10, 10, 0.01)