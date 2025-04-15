import numpy as np
import matplotlib.pyplot as plt

def natural_cubic_spline(x, y):
    n = len(x) - 1
    h = np.diff(x)
    alpha = np.zeros(n)
    
    # Вычисляем alpha
    for i in range(1, n):
        alpha[i] = 3/h[i]*(y[i+1]-y[i]) - 3/h[i-1]*(y[i]-y[i-1])
    
    # Инициализируем матрицу для решения системы уравнений
    l = np.zeros(n+1)
    mu = np.zeros(n)
    z = np.zeros(n+1)
    l[0] = 1
    
    # Прямой ход метода прогонки
    for i in range(1, n):
        l[i] = 2*(x[i+1]-x[i-1]) - h[i-1]*mu[i-1]
        mu[i] = h[i]/l[i]
        z[i] = (alpha[i]-h[i-1]*z[i-1])/l[i]
    
    l[n] = 1
    z[n] = 0
    c = np.zeros(n+1)
    b = np.zeros(n)
    d = np.zeros(n)
    
    # Обратный ход метода прогонки
    for j in range(n-1, -1, -1):
        c[j] = z[j] - mu[j]*c[j+1]
        b[j] = (y[j+1]-y[j])/h[j] - h[j]*(c[j+1]+2*c[j])/3
        d[j] = (c[j+1]-c[j])/(3*h[j])
    
    # Создаем список коэффициентов для каждого интервала
    spline_coeffs = []
    for i in range(n):
        spline_coeffs.append([y[i], b[i], c[i], d[i]])
    
    return spline_coeffs

def evaluate_spline(x, y, coeffs, x_val):
    n = len(x) - 1
    # Находим правильный интервал
    for i in range(n):
        if x[i] <= x_val <= x[i+1]:
            h = x_val - x[i]
            a, b, c, d = coeffs[i]
            return a + b*h + c*h**2 + d*h**3
    return y[-1]  # Если x_val за пределами, возвращаем последнее значение

def plot_spline(x_vals, y_vals, coeffs):
    x_plot = np.linspace(np.min(x_vals), np.max(x_vals), 300)
    y_plot = np.array([evaluate_spline(x_vals, y_vals, coeffs, x) for x in x_plot])
    
    plt.figure(figsize=(10, 6))
    plt.axhline(0, color='black', linewidth=1.3, linestyle='-')  
    plt.axvline(0, color='black', linewidth=1.3, linestyle='-')
    plt.plot(x_vals, y_vals, 'ro', markersize=8, label='Узловые точки')
    plt.plot(x_plot, y_plot, color='pink', linewidth=2, label='Кубический сплайн')
    plt.xlabel('x', fontsize=14)
    plt.ylabel('y', fontsize=14)
    plt.title('Кубический сплайн', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()

def print_coefficients(x_vals, coeffs):
    print("\nКоэффициенты сплайна:")
    for i in range(len(x_vals) - 1):
        print(f"Интервал [{x_vals[i]}, {x_vals[i+1]}]:")
        print(f" a = {coeffs[i][0]:.3f}")
        print(f" b = {coeffs[i][1]:.3f}")
        print(f" c = {coeffs[i][2]:.3f}")
        print(f" d = {coeffs[i][3]:.3f}")

def print_values(x_vals, y_vals, coeffs):
    print("\nЗначения сплайна в узловых точках:")
    spline_values = np.array([evaluate_spline(x_vals, y_vals, coeffs, x) for x in x_vals])
    print("Вычисленные:", np.round(spline_values, 3))
    print("Исходные:", y_vals)
    print("Разница:", np.round(np.abs(spline_values - y_vals), 3))

# Исходные данные
x_vals = np.array([0, 0.5, 1, 1.5])
y_vals = np.array([-1.86, -2.15, -2.57, -3.25])

# Вычисляем коэффициенты сплайна
spline_coeffs = natural_cubic_spline(x_vals, y_vals)

# Выводим результаты
print_coefficients(x_vals, spline_coeffs)
print_values(x_vals, y_vals, spline_coeffs)
plot_spline(x_vals, y_vals, spline_coeffs)