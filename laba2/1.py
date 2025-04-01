import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return 2 * np.sin(x + 0.5) - 1.5 + x

def phi(x):
    return (x + 1.5 - 2 * np.sin(x + 0.5)) / 2

def phi_derivative(x):
    return (1 - 2 * np.cos(x + 0.5)) / 2

def simple_iteration_method(phi, a, b, epsilon, max_iterations=1000):
    x_prev = (a + b) / 2  # Начальное приближение - середина диапазона
    iterations = 0
    iteration_data = []
    
    print("\nПроцесс итераций:")
    print(f"{'Шаг':<5} | {'Xn':<6} | {'Xn+1':<6} | {'|Xn-Xn+1|':<6} | Точность")
    print("-" * 65)
    
    while True:
        x_next = phi(x_prev)
        iterations += 1
        difference = abs(x_next - x_prev)
        
        # Формируем строку для вывода
        comparison = "✓" if difference < epsilon else "✗"
        print(f"{iterations:<5} | {x_prev:.{abs(int(np.log10(epsilon))) + 1}f} | {x_next:.{abs(int(np.log10(epsilon))) + 1}f} | {difference:.{abs(int(np.log10(epsilon))) + 1}f} | {comparison}")
        
        # Проверка точности
        if difference < epsilon:
            break
            
        if iterations >= max_iterations:
            print(f"\nДостигнуто максимальное количество итераций ({max_iterations}).")
            break
            
        x_prev = x_next
    
    return x_next, iterations

def plot_initial_function():
    # График исходной функции на [-5, 5]
    x_wide = np.linspace(-5, 5, 1000)
    plt.figure(figsize=(10, 5))
    plt.plot(x_wide, f(x_wide), label='f(x) = 2*sin(x+0.5) - 1.5 + x')
    plt.axhline(0, color='black', linewidth=0.5)
    plt.title('График исходной функции f(x)')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_iteration_details():
    # Ввод данных
    a = float(input("Введите начало диапазона (A): "))
    b = float(input("Введите конец диапазона (B): "))
    epsilon = float(input("Введите точность (например, 0.0001): "))
    
    display_digits = abs(int(np.log10(epsilon))) + 1  # Точность вывода
    
    # График φ(x) с линиями A и B
    x_fine = np.linspace(a, b, 1000)
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(x_fine, phi(x_fine), label='φ(x) = 1.5 - 2*sin(x+0.5)')
    plt.axhline(a, color='red', linestyle='--', alpha=0.5, label=f'A = {a}')
    plt.axhline(b, color='blue', linestyle='--', alpha=0.5, label=f'B = {b}')
    plt.axvline(a, color='red', linestyle='--', alpha=0.3)
    plt.axvline(b, color='blue', linestyle='--', alpha=0.3)
    plt.title('График φ(x) и границы A, B')
    plt.xlabel('x')
    plt.ylabel('φ(x)')
    plt.legend()
    plt.grid(True)
    
    # График φ'(x) с линиями y=±1
    plt.subplot(1, 2, 2)
    plt.plot(x_fine, phi_derivative(x_fine), label="φ'(x) = -2*cos(x+0.5)")
    plt.axhline(1, color='green', linestyle='--', alpha=0.5, label='y = 1')
    plt.axhline(-1, color='green', linestyle='--', alpha=0.5, label='y = -1')
    plt.axvline(a, color='red', linestyle='--', alpha=0.3)
    plt.axvline(b, color='blue', linestyle='--', alpha=0.3)
    plt.title('График производной φ\'(x)')
    plt.xlabel('x')
    plt.ylabel("φ'(x)")
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # Проверка условия сходимости |φ'(x)| < 1
    if np.any(np.abs(phi_derivative(x_fine)) >= 1):
        print("\nПредупреждение: |φ'(x)| >= 1 на части интервала. Метод может не сходиться.")
    else:
        print("\nУсловие сходимости |φ'(x)| < 1 выполняется на всём интервале.")
    
    # Решение методом простых итераций
    solution, iterations = simple_iteration_method(phi, a, b, epsilon)
    
    # Вывод результата
    print(f"\nНайденное решение: x = {solution:.{display_digits}f}")

if __name__ == "__main__":
    print("Решение уравнения 2*sin(x+0.5) = 1.5 - x методом простых итераций")
    plot_initial_function()
    plot_iteration_details()