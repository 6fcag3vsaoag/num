import numpy as np
import pandas as pd

# Заданные параметры
a, b = 1.3, 2.3
dx = 0.1

# Определение функции
def y(x):
    return x**2 - np.log(x + 1)

# Аналитические производные
def analytical_first_derivative(x):
    return 2*x - 1 / (x + 1)

def analytical_second_derivative(x):
    return 2 + 1 / (x + 1)**2

# Численное дифференцирование
def first_derivative(y_values, x_values, dx):
    return np.gradient(y_values, dx)

def second_derivative(y_values, x_values, dx):
    return np.gradient(np.gradient(y_values, dx), dx)

# Узлы
x_values = np.arange(a, b + dx, dx)
y_values = y(x_values)

# Вычисление производных
y_prime_numerical = first_derivative(y_values, x_values, dx)
y_double_prime_numerical = second_derivative(y_values, x_values, dx)

y_prime_analytical = analytical_first_derivative(x_values)
y_double_prime_analytical = analytical_second_derivative(x_values)

# Таблица для первых производных
data_first_derivative = {
    "x": x_values,
    "y'(численное)": y_prime_numerical,
    "y'(аналитическое)": y_prime_analytical,
    "Разница y'": np.abs(y_prime_numerical - y_prime_analytical)
}
table_first_derivative = pd.DataFrame(data_first_derivative)

# Таблица для вторых производных
data_second_derivative = {
    "x": x_values,
    "y''(численное)": y_double_prime_numerical,
    "y''(аналитическое)": y_double_prime_analytical,
    "Разница y''": np.abs(y_double_prime_numerical - y_double_prime_analytical)
}
table_second_derivative = pd.DataFrame(data_second_derivative)

# Вывод таблиц
print("Таблица первых производных:")
print(table_first_derivative.to_string(index=False))

print("\nТаблица вторых производных:")
print(table_second_derivative.to_string(index=False))
