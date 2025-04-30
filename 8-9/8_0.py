import numpy as np

# Параметры задачи
a = -0.2
b = 0.8
dx = 0.1

# Функция y(x)
def y(x):
    return 2 * np.sin(x + 0.5) + x

# Первая производная y'(x) (аналитическая)
def y_prime_analytical(x):
    return 2 * np.cos(x + 0.5) + 1

# Вторая производная y''(x) (аналитическая)
def y_double_prime_analytical(x):
    return -2 * np.sin(x + 0.5)

# Численная первая производная (центральная разность)
def y_prime_numerical(x, dx):
    return (y(x + dx) - y(x - dx)) / (2 * dx)

# Численная вторая производная (центральная разность)
def y_double_prime_numerical(x, dx):
    return (y(x + dx) - 2 * y(x) + y(x - dx)) / (dx ** 2)

# Создаем массив значений x
x_values = np.arange(a, b + dx, dx)

# Таблица для первой производной
print("Таблица первых производных:")
print("{:<10} {:<25} {:<25} {:<15}".format("x", "y'(численное)", "y'(аналитическое)", "Разница y'"))
for x in x_values:
    # Численное значение первой производной
    y_prime_num = y_prime_numerical(x, dx)
    # Аналитическое значение первой производной
    y_prime_anal = y_prime_analytical(x)
    # Разница
    diff = abs(y_prime_num - y_prime_anal)
    print(f"{x:<10.1f} {y_prime_num:<25.6f} {y_prime_anal:<25.6f} {diff:<15.6f}")

# Таблица для второй производной
print("\nТаблица вторых производных:")
print("{:<10} {:<25} {:<25} {:<15}".format("x", "y''(численное)", "y''(аналитическое)", "Разница y''"))
for x in x_values:
    # Численное значение второй производной
    y_double_prime_num = y_double_prime_numerical(x, dx)
    # Аналитическое значение второй производной
    y_double_prime_anal = y_double_prime_analytical(x)
    # Разница
    diff = abs(y_double_prime_num - y_double_prime_anal)
    print(f"{x:<10.1f} {y_double_prime_num:<25.6f} {y_double_prime_anal:<25.6f} {diff:<15.6f}")