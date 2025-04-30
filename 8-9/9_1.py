import numpy as np

# Функция под интегралом
def f(x):
    return 1 / np.log(x)

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

# Итеративные методы
def iterative(method_func, a, b, e):
    n = 4
    previous_result = None
    history = []
    
    while True:
        result, h, x_values, y_values = method_func(a, b, n)
        diff = abs(result - previous_result) if previous_result is not None else None
        history.append((n, result, diff))
        
        if previous_result is not None and abs(result - previous_result) < e:
            break
        
        previous_result = result
        n *= 2
    
    return history, n, result

# Главная функция
def main():
    # Границы интегрирования
    a = 2
    b = 3

    # Ввод точности
    e = float(input('Введите требуемую точность e: '))

    # Определяем количество знаков после запятой для вывода
    precision = int(-np.floor(np.log10(e)))  # например, e=0.001 -> 3 знака
    output_precision = precision + 1         # выводим на 1 знак больше

    # Аналитическое значение (примерное)
    I_analytical = 1.1184248145497  

    # Форматирование ширины столбцов
    width_n = 6
    width_I = output_precision + 4
    width_diff = output_precision + 4

    # Формат строки для вывода
    format_str = f"{{:<{width_n}}} | {{:<{width_I}}} | {{:<{width_diff}}}"

    # Печать таблицы
    def print_table(history, final_result, method_name):
        print(f"\nМетод: {method_name}")
        print(format_str.format("n", "I", "ΔI"))
        for entry in history:
            n, I, diff = entry
            diff_str = f"{diff:.{output_precision}f}" if diff is not None else "nan"
            print(format_str.format(n, f"{I:.{output_precision}f}", diff_str))
        print(f"Количество разбиений: {history[-1][0]}")
        print(f"Итоговый интеграл: I = {final_result:.{output_precision}f}")
        print(f"Аналитический результат: I = {I_analytical}")

    # Метод левых прямоугольников
    history_rect, n_rect, result_rect = iterative(lambda a,b,n: compute_left_rect(a,b,n), a, b, e)
    print_table(history_rect, result_rect, "Левые прямоугольники")

    # Метод трапеций
    history_trap, n_trap, result_trap = iterative(lambda a,b,n: compute_trapezoidal(a,b,n), a, b, e)
    print_table(history_trap, result_trap, "Трапеции")

    # Метод Симпсона
    history_simp, n_simp, result_simp = iterative(lambda a,b,n: compute_simpson(a,b,n), a, b, e)
    print_table(history_simp, result_simp, "Симпсона")

# Запуск программы
if __name__ == "__main__":
    main()