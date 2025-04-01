import numpy as np
import matplotlib.pyplot as plt

# Определение системы уравнений
def f1(x, y):
    return np.cos(x + 0.5) + y - 1

def f2(x, y):
    return np.sin(y) - 2 * x - 2

# Преобразование системы для метода итераций: x = g1(x, y), y = g2(x, y)
def g1(x, y):
    return (np.sin(y) - 2) / 2

def g2(x, y):
    return 1 - np.cos(x + 0.5)

# Частные производные для проверки условий сходимости
def dg1_dx(x, y):
    return 0  # g1 не зависит от x

def dg1_dy(x, y):
    return np.cos(y) / 2

def dg2_dx(x, y):
    return np.sin(x + 0.5)

def dg2_dy(x, y):
    return 0  # g2 не зависит от y

# Функция для построения графиков
def plot_system():
    x_vals = np.linspace(-4, 4, 500)
    y_vals = np.linspace(-3, 3, 500)
    X, Y = np.meshgrid(x_vals, y_vals)
    
    # Вычисление значений функций
    F1 = np.cos(X + 0.5) + Y - 1
    F2 = np.sin(Y) - 2 * X - 2
    
    plt.figure(figsize=(10, 6))
    plt.contour(X, Y, F1, levels=[0], colors='blue', label="cos(x+0.5)+y=1")
    plt.contour(X, Y, F2, levels=[0], colors='red', label="sin(y)-2x=2")
    plt.axhline(0, color='black', linewidth=0.5, linestyle='--')
    plt.axvline(0, color='black', linewidth=0.5, linestyle='--')
    plt.title("Графики функций системы уравнений")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid()
    plt.legend(["cos(x+0.5)+y=1", "sin(y)-2x=2"])
    plt.show()

# Основная программа
if __name__ == "__main__":
    # Шаг 1: Построение графиков системы уравнений
    print("Графики системы уравнений на интервале [-5, 5]:")
    plot_system()
    
    # Шаг 2: Ввод начальной точки и точности
    x_n = float(input("Введите начальное приближение для x: "))
    y_n = float(input("Введите начальное приближение для y: "))
    epsilon = float(input("Введите точность (например, 0.001): "))
    
    if epsilon <= 0:
        raise ValueError("Точность должна быть положительным числом.")
    
    # Количество знаков после запятой для вывода
    precision_digits = int(np.ceil(-np.log10(epsilon))) + 1
    
    # Шаг 3: Проверка условий сходимости
    print("\nПроверка условий сходимости в начальной точке:")
    dg1_dx_val = abs(dg1_dx(x_n, y_n))
    dg1_dy_val = abs(dg1_dy(x_n, y_n))
    dg2_dx_val = abs(dg2_dx(x_n, y_n))
    dg2_dy_val = abs(dg2_dy(x_n, y_n))
    
    print(f"Производная d(g1)/dx = {dg1_dx_val:.{precision_digits}f} (по модулю {'меньше' if dg1_dx_val < 1 else 'больше или равно'} 1)")
    print(f"Производная d(g1)/dy = {dg1_dy_val:.{precision_digits}f} (по модулю {'меньше' if dg1_dy_val < 1 else 'больше или равно'} 1)")
    print(f"Производная d(g2)/dx = {dg2_dx_val:.{precision_digits}f} (по модулю {'меньше' if dg2_dx_val < 1 else 'больше или равно'} 1)")
    print(f"Производная d(g2)/dy = {dg2_dy_val:.{precision_digits}f} (по модулю {'меньше' if dg2_dy_val < 1 else 'больше или равно'} 1)")
    
    if max(dg1_dx_val, dg1_dy_val, dg2_dx_val, dg2_dy_val) >= 1:
        raise ValueError("Условия сходимости не выполнены. Попробуйте другую начальную точку.")
    
    # Шаг 4: Итерационный процесс
    iteration = 0
    print("\nn | Xn          | Yn          | X(n+1)      | Y(n+1)      | |Xn - X(n+1)| | Точность | |Yn - Y(n+1)|")
    print("-" * 120)
    
    while True:
        iteration += 1
        x_next = g1(x_n, y_n)
        y_next = g2(x_n, y_n)
        
        diff_x = abs(x_n - x_next)
        diff_y = abs(y_n - y_next)
        
        # Вывод информации о текущей итерации
        comparison_x = "меньше" if diff_x < epsilon else "больше"
        comparison_y = "меньше" if diff_y < epsilon else "больше"
        print(f"{iteration:^3} | {x_n:.{precision_digits}f} | {y_n:.{precision_digits}f} | {x_next:.{precision_digits}f} | {y_next:.{precision_digits}f} | {diff_x:.{precision_digits}f} | {comparison_x:^8} | {diff_y:.{precision_digits}f} | {comparison_y}")
        
        # Проверка условия остановки
        if diff_x < epsilon and diff_y < epsilon:
            break
        
        x_n, y_n = x_next, y_next
    
    # Шаг 5: Вывод результата
    print(f"\nСНУ имеет решение: x = {x_next:.{precision_digits}f}, y = {y_next:.{precision_digits}f}")