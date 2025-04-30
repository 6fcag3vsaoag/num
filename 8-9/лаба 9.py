import numpy as np
import pandas as pd
import math

# Функция под интегралом
def f(x):
    return x * np.exp(-x**2)

# Границы интегрирования
a = 0
b = 1

# Ввод требуемой точности
e = float(input('Введите требуемую точность e: '))

precision = int(-np.floor(np.log10(e))) + 1

# Метод левых прямоугольников
def compute_left_rect(a, b, n):
    h = (b - a) / n
    x_values = [a + k * h for k in range(n)]
    y_values = [f(x) for x in x_values]
    integral = h * sum(y_values)
    return integral, h, x_values, y_values

# Метод трапеций
def compute_trapezoidal(a, b, t):
    h = (b - a) / t
    x_values = [a + k * h for k in range(t + 1)]
    y_values = [f(x) for x in x_values]
    integral = h * (y_values[0] + 2 * sum(y_values[1:-1]) + y_values[-1]) / 2
    return integral, h, x_values, y_values

# Метод Симпсона
def compute_simpson(a, b, c):
    if c % 2 != 0:
        raise ValueError("Метод Симпсона требует четное количество разбиений n.")
    
    h = (b - a) / c
    x_values = [a + k * h for k in range(c + 1)]
    y_values = [f(x) for x in x_values]
    integral = (h / 3) * (y_values[0] + y_values[-1] + 
                          4 * sum(y_values[1:-1:2]) + 
                          2 * sum(y_values[2:-1:2]))
    return integral, h, x_values, y_values

# Итеративный процесс для левых прямоугольников
def iterative_left_rect():
    n = 4
    previous_result = None
    history = []
    
    while True:
        result, h, x_values, y_values = compute_left_rect(a, b, n)
        diff = abs(result - previous_result) if previous_result is not None else None
        history.append((n, result, diff))
        
        if previous_result is not None and abs(result - previous_result) < e:
            break
        
        previous_result = result
        n *= 2
    
    return history, n, result

# Итеративный процесс для трапеций
def iterative_trapezoidal():
    t = 4
    previous_result = None
    history = []
    
    while True:
        result, h, x_values, y_values = compute_trapezoidal(a, b, t)
        diff = abs(result - previous_result) if previous_result is not None else None
        history.append((t, result, diff))
        
        if previous_result is not None and abs(result - previous_result) < e:
            break
        
        previous_result = result
        t *= 2
    
    return history, t, result

# Итеративный процесс для Симпсона
def iterative_simpson():
    c = 4
    previous_result = None
    history = []
    
    while True:
        result, h, x_values, y_values = compute_simpson(a, b, c)
        diff = abs(result - previous_result) if previous_result is not None else None
        history.append((c, result, diff))
        
        if previous_result is not None and abs(result - previous_result) < e:
            break
        
        previous_result = result
        c *= 2
    
    return history, c, result

# Вычисление интеграла методом левых прямоугольников
history, n, final_result = iterative_left_rect()
print(f'\nМетод левых прямоугольников')
print(f'Количество разбиений: {n}')
history_table = pd.DataFrame(history, columns=['n', 'I', 'ΔI'])
history_table['I'] = history_table['I'].map(lambda x: f"{x:.4f}")
history_table['ΔI'] = history_table['ΔI'].map(lambda x: f"{x:.4f}" if x is not None else '-')
print(history_table.to_string(index=False))
print(f'Итоговый интеграл: I = {final_result:.{precision}f}')
print(f"\nОпределённый интеграл аналитическим способом равен: 0.3160602794")

# Вычисление интеграла методом трапеций
history, t, final_result = iterative_trapezoidal()
print(f'\nМетод трапеций')
print(f'Количество разбиений: {t}')
history_table = pd.DataFrame(history, columns=['n', 'I', 'ΔI'])
history_table['I'] = history_table['I'].map(lambda x: f"{x:.4f}")
history_table['ΔI'] = history_table['ΔI'].map(lambda x: f"{x:.4f}" if x is not None else '-')
print(history_table.to_string(index=False))
print(f'Итоговый интеграл: I = {final_result:.{precision}f}')
print(f"\nОпределённый интеграл аналитическим способом равен: 0.3160602794")

# Вычисление интеграла методом Симпсона
history, c, final_result = iterative_simpson()
print(f'\nМетод Симпсона')
print(f'Количество разбиений: {c}')
history_table = pd.DataFrame(history, columns=['n', 'I', 'ΔI'])
history_table['I'] = history_table['I'].map(lambda x: f"{x:.4f}")
history_table['ΔI'] = history_table['ΔI'].map(lambda x: f"{x:.4f}" if x is not None else '-')
print(history_table.to_string(index=False))
print(f'Итоговый интеграл: I = {final_result:.{precision}f}')
print(f"\nОпределённый интеграл аналитическим способом равен: 0.3160602794")
