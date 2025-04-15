import numpy as np
import matplotlib.pyplot as plt

def print_table(x_vals, y_vals):
    print("Исходная таблица:")
    for i in range(len(x_vals)):
        print(f"x_{i} = {x_vals[i]:.2f}, y_{i} = {y_vals[i]:.2f}")
    print()

def divided_diff(x_vals, y_vals):
    n = len(x_vals)
    diff_table = np.zeros((n, n))
    diff_table[:, 0] = y_vals

    for j in range(1, n):
        for i in range(n-j):
            diff_table[i][j] = (diff_table[i+1][j-1] - diff_table[i][j-1]) / (x_vals[i+j] - x_vals[i])
    return diff_table[0]

def get_newton_poly(x_vals, coef):
    n = len(coef)
    poly = np.polynomial.Polynomial([0.])
    
    for i in range(n):
        term = np.polynomial.Polynomial([coef[i]])
        for j in range(i):
            term = term * np.polynomial.Polynomial([-x_vals[j], 1])
        poly = poly + term
    
    return poly

def get_lagrange_poly(x_vals, y_vals):
    n = len(x_vals)
    poly = np.polynomial.Polynomial([0.])
    
    for i in range(n):
        numerator = np.polynomial.Polynomial([1.])
        denominator = 1.0
        
        for j in range(n):
            if i != j:
                numerator = numerator * np.polynomial.Polynomial([-x_vals[j], 1])
                denominator *= (x_vals[i] - x_vals[j])
        
        term = numerator * (y_vals[i] / denominator)
        poly = poly + term
    
    return poly

def format_poly(poly):
    coeffs = poly.convert().coef
    terms = []
    
    for power, coeff in enumerate(coeffs):
        if abs(coeff) < 1e-10:  # Игнорируем очень маленькие коэффициенты
            continue
            
        if power == 0:
            term = f"{coeff:.3f}"
        else:
            if abs(coeff - 1) < 1e-10:
                term = ""
            elif abs(coeff + 1) < 1e-10:
                term = "-"
            else:
                term = f"{coeff:.3f}"
            
            term += "x" + ("^"+str(power) if power > 1 else "")
        
        terms.append(term)
    
    # Собираем члены от старшей степени к младшей
    terms.reverse()
    
    # Обрабатываем знаки
    poly_str = terms[0]
    for term in terms[1:]:
        if term.startswith('-'):
            poly_str += " - " + term[1:]
        else:
            poly_str += " + " + term
    
    return poly_str.replace("+ -", "- ")

def plot_graphs(x_vals, y_vals, newton_poly, lagrange_poly):
    x_plot = np.linspace(min(x_vals)-0.2, max(x_vals)+0.2, 500)
    
    y_newton = newton_poly(x_plot)
    y_lagrange = lagrange_poly(x_plot)

    plt.figure(figsize=(10, 6))
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)
    plt.plot(x_plot, y_newton, label="Полином Ньютона", color='blue')
    plt.plot(x_plot, y_lagrange, '--', label="Полином Лагранжа", color='red', alpha=0.7)
    plt.scatter(x_vals, y_vals, color='green', label="Узлы интерполяции", zorder=3)
    plt.title("Интерполяционные полиномы", fontsize=14)
    plt.xlabel("x", fontsize=12)
    plt.ylabel("y", fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True)
    plt.show()

# Исходные данные
x_vals = np.array([0, 0.5, 1, 1.5])
y_vals = np.array([-1.86, -2.15, -2.57, -3.25])

# 1. Вывод исходной таблицы
print_table(x_vals, y_vals)

# 2. Вычисление и вывод полинома Ньютона
newton_coef = divided_diff(x_vals, y_vals)
newton_poly = get_newton_poly(x_vals, newton_coef)
print("Полином Ньютона:")
print("N(x) =", format_poly(newton_poly))
print()

# 3. Вычисление и вывод полинома Лагранжа
lagrange_poly = get_lagrange_poly(x_vals, y_vals)
print("Полином Лагранжа:")
print("L(x) =", format_poly(lagrange_poly))
print()

# 4. Построение графика
print("График интерполяционных полиномов:")
plot_graphs(x_vals, y_vals, newton_poly, lagrange_poly)