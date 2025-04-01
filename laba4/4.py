import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
import math

x_sym = sp.Symbol('x')
y_sym = sp.Symbol('y')

# Исходные уравнения системы
f1 = sp.cos(x_sym + 0.5) + y_sym - 1
f2 = sp.sin(y_sym) - 2*x_sym - 2

# Якобиан системы
J = sp.Matrix([[sp.diff(f1, x_sym), sp.diff(f1, y_sym)],
               [sp.diff(f2, x_sym), sp.diff(f2, y_sym)]])

# Обратный якобиан
J_inv = J.inv()

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
    x_sol = float(x_sol)
    y_sol = float(y_sol)
    
    x_vals = np.linspace(x_sol - zoom_scale, x_sol + zoom_scale, 1000)
    y_vals = np.linspace(y_sol - zoom_scale, y_sol + zoom_scale, 1000)

    f1_np = lambda x, y: np.cos(x + 0.5) + y - 1
    f2_np = lambda x, y: np.sin(y) - 2*x - 2

    x_grid, y_grid = np.meshgrid(x_vals, y_vals)
    res1 = f1_np(x_grid, y_grid)
    res2 = f2_np(x_grid, y_grid)

    plt.figure(figsize=(10, 8))
    
    cs1 = plt.contour(x_grid, y_grid, res1, levels=[0], colors='purple', linewidths=2)
    cs2 = plt.contour(x_grid, y_grid, res2, levels=[0], colors='red', linewidths=2)
    
    plt.scatter([x_sol], [y_sol], color='lime', s=100, edgecolor='black', zorder=5)
    plt.axhline(y_sol, color='black', linestyle=':', alpha=0.3)
    plt.axvline(x_sol, color='black', linestyle=':', alpha=0.3)
    
    plt.gca().set_aspect('equal')
    plt.gca().xaxis.set_major_formatter(plt.FormatStrFormatter('%.8f'))
    plt.gca().yaxis.set_major_formatter(plt.FormatStrFormatter('%.8f'))
    plt.xticks(rotation=45)
    
    plt.legend([plt.Line2D([], [], color='purple', linewidth=2),
                plt.Line2D([], [], color='red', linewidth=2)],
               [f'cos(x+0.5)+y-1=0 (x={x_sol:.8f})',
                f'sin(y)-2x-2=0 (y={y_sol:.8f})'],
               loc='upper right')
    
    plt.title(f"Ультра-приближение к решению (масштаб: {zoom_scale:.1e})")
    plt.grid(True, which='both', linestyle=':', alpha=0.5)
    plt.tight_layout()
    plt.show()

def eps(e):
    """Определение количества знаков после запятой для точности"""
    e = float(e)
    if e == 0:
        return 1
    s = "{:.15f}".format(abs(e)).rstrip('0')
    if '.' in s:
        return len(s.split('.')[1])
    return 0

def check_newton_convergence(x0, y0):
    """Проверка условий сходимости метода Ньютона"""
    # Подставляем начальное приближение в якобиан
    J_val = J.subs({x_sym: x0, y_sym: y0})
    
    # 1. Проверяем, что определитель якобиана не нулевой
    det_J = float(J_val.det())
    if abs(det_J) < 1e-10:
        print(f"⚠️ Якобиан вырожден (det(J) = {det_J:.2e}). Метод может не сойтись.")
        return False
    
    # 2. Проверяем, что матрица Якоби обратима (дополнительная проверка)
    try:
        J_inv = J_val.inv()
    except:
        print("⚠️ Якобиан необратим. Метод не сойдётся.")
        return False
    
    # 3. Проверяем, что начальная точка близка к решению (эвристика)
    F_val = sp.Matrix([f1.subs({x_sym: x0, y_sym: y0}), 
                       f2.subs({x_sym: x0, y_sym: y0})])
    norm_F = max(abs(F_val[0]), abs(F_val[1]))
    if norm_F > 10:
        print(f"⚠️ Начальная точка далеко от решения (||F|| = {norm_F:.2f}). Возможна расходимость.")
    
    return True

def newton_method(x0, y0, epsilon):
    """Метод Ньютона с упрощенным выводом о точности"""
    x, y = x0, y0
    i = 0
    dec_point = eps(epsilon)
    
    print(f"{'Шаг':<5} | {'x_old':<10} | {'x_new':<10} | {'y_old':<10} | {'y_new':<10} | {'Δx':<10} | {'Δy':<10} | {'Точность':<10}")
    print("-" * 85)
    
    while True:
        # Вычисление функций и якобиана
        F = sp.Matrix([[f1.subs({x_sym: x, y_sym: y})], 
                      [f2.subs({x_sym: x, y_sym: y})]])
        J_current = J.subs({x_sym: x, y_sym: y})
        
        # Вычисление приращения
        delta = J_current.LUsolve(-F)
        delta_x, delta_y = float(delta[0]), float(delta[1])
        x_new, y_new = x + delta_x, y + delta_y
        
        # Определение статуса точности
        precision_status = "ДА" if abs(delta_x) < epsilon and abs(delta_y) < epsilon else "НЕТ"
        
        # Вывод итерации
        print(f"{i+1:<5} | {x:<10.{dec_point}f} | {x_new:<10.{dec_point}f} | "
              f"{y:<10.{dec_point}f} | {y_new:<10.{dec_point}f} | "
              f"{delta_x:<10.{dec_point}f} | {delta_y:<10.{dec_point}f} | "
              f"{precision_status:<10}")
        
        # Проверка условия останова
        if precision_status == "ДА":
            print(f"\nРешение найдено с точностью {epsilon:.{dec_point}f}")
            break
            
        x, y = x_new, y_new
        i += 1
    
    return x, y

# Основная программа
initial_system_graph()

x0 = float(input("Введите начальное приближение x: "))
y0 = float(input("Введите начальное приближение y: "))
epsilon = float(input("Введите точность epsilon: "))
dec_point = eps(epsilon)

if check_newton_convergence(x0, y0):
    x_res, y_res = newton_method(x0, y0, epsilon)
else:
    print("Попробуйте другое начальное приближение.")

print(f"\nРезультат:")
print(f"x = {float(x_res):.{dec_point}f}")
print(f"y = {float(y_res):.{dec_point}f}")

#solution_graph(x_res, y_res)