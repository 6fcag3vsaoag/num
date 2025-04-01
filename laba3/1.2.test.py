import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
import math

x_sym = sp.Symbol('x')
y_sym = sp.Symbol('y')

# Заданные функции
f1 = sp.cos(x_sym + 0.5) + y_sym - 1
f2 = sp.sin(y_sym) - 2*x_sym - 2

# Функции для метода простых итераций
func = sp.sin(y_sym)/2 - 1  # x = φ1(y)
phi = 1 - sp.cos(x_sym + 0.5)  # y = φ2(x)

# Производные для проверки условий сходимости
func_diff_x = sp.diff(func, x_sym)
func_diff_y = sp.diff(func, y_sym)
phi_diff_x = sp.diff(phi, x_sym)
phi_diff_y = sp.diff(phi, y_sym)

def initial_system_graph(a=-5, b=5):
    """График начальной системы уравнений"""
    x_vals = np.linspace(a, b, 400)
    y_vals = np.linspace(a, b, 400)

    f1_np = sp.lambdify((x_sym, y_sym), f1, 'numpy')
    f2_np = sp.lambdify((x_sym, y_sym), f2, 'numpy')

    x_grid, y_grid = np.meshgrid(x_vals, y_vals)
    res1 = f1_np(x_grid, y_grid)
    res2 = f2_np(x_grid, y_grid)

    plt.figure(figsize=(8, 6))
    plt.axhline(0, color='black', linewidth=1.3, linestyle='-')  
    plt.axvline(0, color='black', linewidth=1.3, linestyle='-')
    
    # Рисуем контуры и сохраняем объекты
    cs1 = plt.contour(x_grid, y_grid, res1, levels=[0], colors='purple')
    cs2 = plt.contour(x_grid, y_grid, res2, levels=[0], colors='red')
    
    # Создаем proxy artists для легенды
    proxy1 = plt.Rectangle((0,0), 1, 1, fc='purple')
    proxy2 = plt.Rectangle((0,0), 1, 1, fc='red')
    
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend([proxy1, proxy2], 
              [r'$\cos(x + 0.5) + y - 1 = 0$', 
               r'$\sin(y) - 2x - 2 = 0$'])
    plt.title("Графическое решение системы уравнений")
    plt.grid(True)
    plt.show()

def solution_graph(x_sol, y_sol, zoom_scale=0.001):
    """График решения с ультра-приближением"""
    # Преобразуем в float для гарантии
    x_sol = float(x_sol)
    y_sol = float(y_sol)
    
    # Создаем очень маленькую область вокруг решения
    x_vals = np.linspace(x_sol - zoom_scale, x_sol + zoom_scale, 1000)
    y_vals = np.linspace(y_sol - zoom_scale, y_sol + zoom_scale, 1000)

    # Используем numpy-версии функций
    f1_np = lambda x, y: np.cos(x + 0.5) + y - 1
    f2_np = lambda x, y: np.sin(y) - 2*x - 2

    x_grid, y_grid = np.meshgrid(x_vals, y_vals)
    res1 = f1_np(x_grid, y_grid)
    res2 = f2_np(x_grid, y_grid)

    plt.figure(figsize=(10, 8))
    
    # Рисуем контуры с высокой детализацией
    cs1 = plt.contour(x_grid, y_grid, res1, levels=[0], colors='purple', linewidths=2)
    cs2 = plt.contour(x_grid, y_grid, res2, levels=[0], colors='red', linewidths=2)
    
    # Точка решения (больше размер и контраст)
    plt.scatter([x_sol], [y_sol], color='lime', s=100, edgecolor='black', zorder=5)
    
    # Перекрестие для точной оценки
    plt.axhline(y_sol, color='black', linestyle=':', alpha=0.3)
    plt.axvline(x_sol, color='black', linestyle=':', alpha=0.3)
    
    # Настройка осей
    plt.gca().set_aspect('equal')
    plt.gca().xaxis.set_major_formatter(plt.FormatStrFormatter('%.8f'))
    plt.gca().yaxis.set_major_formatter(plt.FormatStrFormatter('%.8f'))
    plt.xticks(rotation=45)
    
    # Легенда с высокой точностью
    plt.legend([plt.Line2D([], [], color='purple', linewidth=2),
                plt.Line2D([], [], color='red', linewidth=2)],
               [f'cos(x+0.5)+y-1=0 (x={x_sol:.8f})',
                f'sin(y)-2x-2=0 (y={y_sol:.8f})'],
               loc='upper right')
    
    plt.title(f"Ультра-приближение к решению (масштаб: {zoom_scale:.1e})")
    plt.grid(True, which='both', linestyle=':', alpha=0.5)
    plt.tight_layout()
    plt.show()


def convergence_conditions(x_0, y_0):
    """Проверка условий сходимости"""
    if func_diff_x.subs({x_sym: x_0}) > 1 or func_diff_y.subs({y_sym: y_0}) > 1:
        return False
    elif phi_diff_x.subs({x_sym: x_0}) > 1 or phi_diff_y.subs({y_sym: y_0}) > 1:
        return False
    return True

def eps(e):
    """Определение количества знаков после запятой для точности"""
    e = float(e)
    if e == 0:
        return 1
    s = "{:.15f}".format(abs(e)).rstrip('0')
    if '.' in s:
        return len(s.split('.')[1])
    return 0

def simple_iter(x_0, y_0, e):
    """Метод простых итераций"""
    x, y = x_0, y_0
    i = 0
    dec_point = eps(e)
    x_new = func.subs({y_sym: y})
    y_new = phi.subs({x_sym: x})
    delta_x = sp.Abs(x - x_new)
    delta_y = sp.Abs(y - y_new)
    
    print(f"Итерация {i + 1}:")
    print(f"x{i} = {x:.{dec_point}f}")
    print(f"x{i+1} = {x_new:.{dec_point}f}")
    print(f"y{i} = {y:.{dec_point}f}")
    print(f"y{i+1} = {y_new:.{dec_point}f}")
    print(f"Δx = {delta_x:.{dec_point}f}")
    print(f"Δy = {delta_y:.{dec_point}f}\n")

    while delta_x > e or delta_y > e:
        i += 1
        x, y = x_new, y_new
        x_new = func.subs({y_sym: y})
        y_new = phi.subs({x_sym: x})
        delta_x = sp.Abs(x - x_new)
        delta_y = sp.Abs(y - y_new)
        
        print(f"Итерация {i + 1}:")
        print(f"x{i} = {x:.{dec_point}f}")
        print(f"x{i+1} = {x_new:.{dec_point}f}")
        print(f"y{i} = {y:.{dec_point}f}")
        print(f"y{i+1} = {y_new:.{dec_point}f}")
        print(f"Δx = {delta_x:.{dec_point}f}")
        print(f"Δy = {delta_y:.{dec_point}f}\n")

    print(f"Условия точности достигнуты:")
    print(f"Δx = {delta_x:.{dec_point}f} < {e:.{dec_point}f}")
    print(f"Δy = {delta_y:.{dec_point}f} < {e:.{dec_point}f}")
    return x_new, y_new



# Основная программа
initial_system_graph()

x_0 = float(input("Введите начальное приближение x: "))
y_0 = float(input("Введите начальное приближение y: "))
epsilon = float(input("Введите точность epsilon: "))
dec_point = eps(epsilon)

if convergence_conditions(x_0, y_0):
    print("\nУсловия сходимости выполнены:")
    print(f"φ1'(x) = {func_diff_x.subs({x_sym: x_0}):.{dec_point}f} < 1")
    print(f"φ1'(y) = {func_diff_y.subs({y_sym: y_0}):.{dec_point}f} < 1")
    print(f"φ2'(x) = {phi_diff_x.subs({x_sym: x_0}):.{dec_point}f} < 1")
    print(f"φ2'(y) = {phi_diff_y.subs({y_sym: y_0}):.{dec_point}f} < 1\n")
    
    x_res, y_res = simple_iter(x_0, y_0, epsilon)

    print(f"\nРезультат:")
    print(f"x = {x_res:.{dec_point}f}")
    print(f"y = {y_res:.{dec_point}f}")

    #solution_graph(x_res, y_res)
else:
    print('Условия сходимости не выполнены, метод может не сходиться')