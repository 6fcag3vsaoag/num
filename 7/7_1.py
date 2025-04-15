def calculate_coefficients(x_values, y_values):
    n = len(x_values)
    
    # Суммы элементов (предварительные шаги)
    sum_x = sum(x_values)
    sum_y = sum(y_values)
    sum_x_squared = sum(xi * xi for xi in x_values)
    sum_xy = sum(xi * yi for xi, yi in zip(x_values, y_values))
    sum_x_cubed = sum(xi ** 3 for xi in x_values)
    sum_x_fourth = sum(xi ** 4 for xi in x_values)
    sum_x_squared_y = sum((xi ** 2) * yi for xi, yi in zip(x_values, y_values))
    
    print("Промежуточные вычисления:")
    print(f"Сумма X: {sum_x}")
    print(f"Сумма Y: {sum_y}")
    print(f"Сумма X²: {sum_x_squared}")
    print(f"Сумма XY: {sum_xy}")
    print(f"Сумма X³: {sum_x_cubed}")
    print(f"Сумма X⁴: {sum_x_fourth}")
    print(f"Сумма X²Y: {sum_x_squared_y}")
    
    # Матрица системы уравнений
    matrix_A = [
        [n,       sum_x,           sum_x_squared],
        [sum_x,   sum_x_squared,   sum_x_cubed],
        [sum_x_squared, sum_x_cubed, sum_x_fourth]
    ]
    
    vector_B = [sum_y, sum_xy, sum_x_squared_y]
    
    # Функция для нахождения детерминанта матрицы 3х3
    def determinant(matrix):
        return (
            matrix[0][0]*matrix[1][1]*matrix[2][2] +
            matrix[0][1]*matrix[1][2]*matrix[2][0] +
            matrix[0][2]*matrix[1][0]*matrix[2][1] -
            matrix[0][2]*matrix[1][1]*matrix[2][0] -
            matrix[0][1]*matrix[1][0]*matrix[2][2] -
            matrix[0][0]*matrix[1][2]*matrix[2][1]
        )
    
    # Определитель главной матрицы
    det_main = determinant(matrix_A)
    print("\nОпределитель основной матрицы:", det_main)
    
    # Замещение столбцов и вычисление остальных определителей
    matrix_A_a = [row[:] for row in matrix_A]
    matrix_A_a[0][0], matrix_A_a[1][0], matrix_A_a[2][0] = vector_B
    det_a = determinant(matrix_A_a)
    print("Определитель для 'a':", det_a)
    
    matrix_A_b = [row[:] for row in matrix_A]
    matrix_A_b[0][1], matrix_A_b[1][1], matrix_A_b[2][1] = vector_B
    det_b = determinant(matrix_A_b)
    print("Определитель для 'b':", det_b)
    
    matrix_A_c = [row[:] for row in matrix_A]
    matrix_A_c[0][2], matrix_A_c[1][2], matrix_A_c[2][2] = vector_B
    det_c = determinant(matrix_A_c)
    print("Определитель для 'c':", det_c)
    
    # Найдем коэффициенты
    a = det_a / det_main
    b = det_b / det_main
    c = det_c / det_main
    
    return a, b, c


# Данные из столбца №19 (пример ваших данных)
xk = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1,
      1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2]
yk = [-1.89, -2.07, -2.30, -2.26, -2.34, -2.66, -2.88, -2.85, -3.16, -3.49,
      -3.88, -4.22, -4.45, -4.99, -5.36, -5.71, -6.51, -6.76, -7.35, -8.02]

# Расчет коэффициентов
a, b, c = calculate_coefficients(xk, yk)

print("\nИтоговые коэффициенты:")
print(f"a = {a:.4f}, b = {b:.4f}, c = {c:.4f}")