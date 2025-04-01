import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return 2 * np.sin(x + 0.5) - 1.5 + x

def phi(x, omega=1.0):
    """Функция для итераций с параметром релаксации omega"""
    return omega * (1.5 - 2 * np.sin(x + 0.5)) + (1 - omega) * x

def phi_derivative(x, omega=1.0):
    """Производная функции phi с параметром релаксации"""
    return omega * (-2 * np.cos(x + 0.5)) + (1 - omega)

def find_optimal_omega(a, b, epsilon):
    """Автоматический подбор оптимального параметра релаксации"""
    omega_values = np.linspace(0.1, 1.9, 19)
    best_omega = 1.0
    min_iterations = float('inf')
    
    # Тестируем разные значения omega
    for omega in omega_values:
        # Проверяем условие сходимости |phi'(x)| < 1
        derivative_values = np.abs(phi_derivative(np.linspace(a, b, 100), omega))
        if np.any(derivative_values >= 1):
            continue
            
        # Пробный запуск метода
        _, iterations = simple_iteration_method(
            lambda x: phi(x, omega), a, b, epsilon, max_iterations=100
        )
        
        if iterations < min_iterations:
            min_iterations = iterations
            best_omega = omega
            
    return best_omega

def simple_iteration_method(phi_func, a, b, epsilon, max_iterations=1000):
    """Универсальный метод простых итераций"""
    x_prev = (a + b) / 2
    iterations = 0
    
    print("\nПроцесс итераций:")
    print(f"{'Шаг':<5} | {'Xn':<10} | {'Xn+1':<10} | {'|Xn-Xn+1|':<12} | Точность")
    print("-" * 65)
    
    while True:
        x_next = phi_func(x_prev)
        iterations += 1
        difference = abs(x_next - x_prev)
        
        precision = "✓" if difference < epsilon else "✗"
        decimals = abs(int(np.log10(epsilon))) + 1
        print(f"{iterations:<5} | {x_prev:<10.{decimals}f} | {x_next:<10.{decimals}f} | "
              f"{difference:<12.{decimals}f} | {precision}")
        
        if difference < epsilon:
            break
        if iterations >= max_iterations:
            print(f"\nДостигнуто максимальное количество итераций ({max_iterations}).")
            break
            
        x_prev = x_next
    
    return x_next, iterations

def plot_relaxation_analysis(a, b):
    """Анализ влияния параметра релаксации"""
    omegas = np.linspace(0.1, 1.9, 19)
    iterations = []
    
    plt.figure(figsize=(10, 5))
    for omega in omegas:
        # Проверяем условие сходимости
        deriv = phi_derivative(np.linspace(a, b, 100), omega)
        if np.all(np.abs(deriv) < 1):
            _, iters = simple_iteration_method(
                lambda x: phi(x, omega), a, b, 1e-6, max_iterations=500
            )
            iterations.append(iters)
        else:
            iterations.append(np.nan)
    
    plt.plot(omegas, iterations, 'o-', label='Число итераций')
    plt.axvline(1.0, color='r', linestyle='--', label='Без релаксации (ω=1)')
    plt.xlabel('Параметр релаксации ω')
    plt.ylabel('Число итераций')
    plt.title('Зависимость скорости сходимости от параметра релаксации')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    return omegas[np.nanargmin(iterations)]

def plot_iteration_details():
    # Ввод данных
    a = float(input("Введите начало диапазона (A): "))
    b = float(input("Введите конец диапазона (B): "))
    epsilon = float(input("Введите точность (например, 0.0001): "))
    
    # Анализ релаксации
    optimal_omega = find_optimal_omega(a, b, epsilon)
    print(f"\nОптимальный параметр релаксации: ω = {optimal_omega:.2f}")
    
    # Визуализация влияния параметра релаксации
    plot_relaxation_analysis(a, b)
    
    # Решение с оптимальным параметром релаксации
    print(f"\nРешение с оптимальным параметром ω = {optimal_omega:.2f}:")
    solution, iterations = simple_iteration_method(
        lambda x: phi(x, optimal_omega), a, b, epsilon
    )
    
    # Вывод результата
    decimals = abs(int(np.log10(epsilon))) + 1
    print(f"\nНайденное решение: x = {solution:.{decimals}f} (за {iterations} итераций)")
    
    # Сравнение с обычным методом (ω=1)
    print("\nСравнение с обычным методом (ω=1.0):")
    _, std_iterations = simple_iteration_method(
        lambda x: phi(x, 1.0), a, b, epsilon
    )
    print(f"\nУскорение: {std_iterations/iterations:.1f} раз")

if __name__ == "__main__":
    print("Решение уравнения 2*sin(x+0.5) = 1.5 - x методом релаксации")
    plot_initial_function()
    plot_iteration_details()