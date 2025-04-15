import matplotlib.pyplot as plt
import numpy as np

def calculate_coefficients(x_values, y_values):
    n = len(x_values)
    
    # Вычисление промежуточных значений для таблицы
    table_data = []
    for k, (xk, yk) in enumerate(zip(x_values, y_values), start=1):
        xk2 = xk * xk
        xk3 = xk ** 3
        xk4 = xk ** 4
        xkyk = xk * yk
        xk2yk = xk2 * yk
        
        table_data.append([
            k, xk, yk, xk2, xk3, xk4, xkyk, xk2yk
        ])
    
    # Вывод таблицы промежуточных вычислений
    print("Таблица промежуточных вычислений:")
    print("k\txk\tyk\txk²\txk³\txk⁴\txkyk\txk²yk")
    print("-" * 90)
    for row in table_data:
        print(f"{row[0]}\t{row[1]:.1f}\t{row[2]:.2f}\t{row[3]:.2f}\t{row[4]:.3f}\t{row[5]:.4f}\t{row[6]:.3f}\t{row[7]:.4f}")
    
    # Вычисление сумм
    sum_x = sum(xk for _, xk, _, _, _, _, _, _ in table_data)
    sum_y = sum(yk for _, _, yk, _, _, _, _, _ in table_data)
    sum_x2 = sum(xk2 for _, _, _, xk2, _, _, _, _ in table_data)
    sum_x3 = sum(xk3 for _, _, _, _, xk3, _, _, _ in table_data)
    sum_x4 = sum(xk4 for _, _, _, _, _, xk4, _, _ in table_data)
    sum_xy = sum(xkyk for _, _, _, _, _, _, xkyk, _ in table_data)
    sum_x2y = sum(xk2yk for _, _, _, _, _, _, _, xk2yk in table_data)
    
    # Вывод итоговой строки
    print("-" * 90)
    print(f"Σ\t{sum_x:.1f}\t{sum_y:.2f}\t{sum_x2:.1f}\t{sum_x3:.1f}\t{sum_x4:.4f}\t{sum_xy:.3f}\t{sum_x2y:.3f}")
    print("\n")
    
    # Вывод системы нормальных уравнений
    print("Система нормальных уравнений:")
    print("⎧ a·Σx⁴ + b·Σx³ + c·Σx² = Σx²y")
    print(f"⎨ a·{sum_x4:.4f} + b·{sum_x3:.1f} + c·{sum_x2:.1f} = {sum_x2y:.3f}")
    print("⎪ a·Σx³ + b·Σx² + c·Σx = Σxy")
    print(f"⎨ a·{sum_x3:.1f} + b·{sum_x2:.1f} + c·{sum_x:.1f} = {sum_xy:.3f}")
    print("⎪ a·Σx² + b·Σx + c·n = Σy")
    print(f"⎩ a·{sum_x2:.1f} + b·{sum_x:.1f} + c·{n} = {sum_y:.2f}")
    print("\n")
    
    # Матрица системы уравнений
    matrix_A = [
        [sum_x4, sum_x3, sum_x2],
        [sum_x3, sum_x2, sum_x],
        [sum_x2, sum_x, n]
    ]
    
    vector_B = [sum_x2y, sum_xy, sum_y]
    
    # Решение системы методом Крамера
    def determinant(matrix):
        return (
            matrix[0][0]*matrix[1][1]*matrix[2][2] +
            matrix[0][1]*matrix[1][2]*matrix[2][0] +
            matrix[0][2]*matrix[1][0]*matrix[2][1] -
            matrix[0][2]*matrix[1][1]*matrix[2][0] -
            matrix[0][1]*matrix[1][0]*matrix[2][2] -
            matrix[0][0]*matrix[1][2]*matrix[2][1]
        )
    
    det_main = determinant(matrix_A)
    matrix_A_a = [row[:] for row in matrix_A]
    matrix_A_a[0][0], matrix_A_a[1][0], matrix_A_a[2][0] = vector_B
    det_a = determinant(matrix_A_a)
    
    matrix_A_b = [row[:] for row in matrix_A]
    matrix_A_b[0][1], matrix_A_b[1][1], matrix_A_b[2][1] = vector_B
    det_b = determinant(matrix_A_b)
    
    matrix_A_c = [row[:] for row in matrix_A]
    matrix_A_c[0][2], matrix_A_c[1][2], matrix_A_c[2][2] = vector_B
    det_c = determinant(matrix_A_c)
    
    a = det_a / det_main
    b = det_b / det_main
    c = det_c / det_main
    
    # Вывод коэффициентов
    print("Итоговые коэффициенты полинома ax² + bx + c:")
    print(f"a = {a:.6f}")
    print(f"b = {b:.6f}")
    print(f"c = {c:.6f}")
    
    # Таблица проверки (невязок)
    print("\nТаблица проверки (невязок):")
    print("k\txk\tyk\tyk(теор)\tНевязка\tКвадрат невязки")
    print("-" * 70)
    
    sum_squared_residuals = 0
    for row in table_data:
        k, xk, yk, _, _, _, _, _ = row
        yk_theor = a*xk**2 + b*xk + c
        residual = yk_theor - yk
        squared_residual = residual**2
        sum_squared_residuals += squared_residual
        
        print(f"{k}\t{xk:.1f}\t{yk:.2f}\t{yk_theor:.6f}\t{residual:.6f}\t{squared_residual:.6f}")
    
    print("-" * 70)
    print(f"Сумма квадратов невязок: {sum_squared_residuals:.6f}")
    
    # Построение графика
    plt.figure(figsize=(10, 6))
    
    # Исходные точки
    plt.scatter(x_values, y_values, color='red', label='Исходные данные', zorder=5)
    
    # Аппроксимирующая кривая
    x_curve = np.linspace(min(x_values), max(x_values), 100)
    y_curve = a*x_curve**2 + b*x_curve + c
    plt.plot(x_curve, y_curve, 'b-', label=f'Аппроксимация: {a:.3f}x² + {b:.3f}x + {c:.3f}')
    
    plt.title('Аппроксимация полиномом второй степени')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid(True)
    plt.legend()
    plt.show()
    
    return a, b, c


# Исходные данные
xk = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1,
      1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2]
yk = [-1.89, -2.07, -2.30, -2.26, -2.34, -2.66, -2.88, -2.85, -3.16, -3.49,
      -3.88, -4.22, -4.45, -4.99, -5.36, -5.71, -6.51, -6.76, -7.35, -8.02]

# Расчет коэффициентов и построение графика
a, b, c = calculate_coefficients(xk, yk)