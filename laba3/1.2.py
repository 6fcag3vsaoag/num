import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
import math


x_sym = sp.Symbol('x')
y_sym = sp.Symbol('y')

func = sp.sin(y_sym)/2 - 1
func_diff_x = sp.diff(func, x_sym)
func_diff_y = sp.diff(func, y_sym)
phi = 1 - sp.cos(x_sym + 0.5)
phi_diff_x = sp.diff(phi, x_sym)
phi_diff_y = sp.diff(phi, y_sym)

def zero_approximation_graph(a = -2, b = 2):
    x_vals = np.linspace(a, b, 400)
    y_vals = np.linspace(a, b, 400)

    f_1 = sp.cos(y_sym + 0.5) + x_sym - 0.8
    f_2 = sp.sin(x_sym) - 2*y_sym - 1.6

    f_1_np = sp.lambdify((x_sym, y_sym), f_1, 'numpy')
    f_2_np = sp.lambdify((x_sym, y_sym), f_2, 'numpy')

    x_grid, y_grid = np.meshgrid(x_vals, y_vals)
    res_1 = f_1_np(x_grid, y_grid)
    res_2 = f_2_np(x_grid, y_grid)


    plt.axhline(0, color='black', linewidth=1.3, linestyle='-')  
    plt.axvline(0, color='black', linewidth=1.3, linestyle='-')
    plt.contour(x_grid, y_grid, res_1, levels = [0], color = 'purple')
    plt.contour(x_grid, y_grid, res_2, levels=[0], colors='red')
    plt.annotate(r'$\cos(y + 0.5) + x - 0.8 = 0$', xy=(-1.5, 0.5), color='purple', fontsize=10)
    plt.annotate(r'$\sin(x) - 2y - 1.6 = 0$', xy=(0.5, -1), color='red', fontsize=10)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.title("Метод итераций для решения системы нелинейных уравнений")
    plt.show()

def convergence_conditions(x_0, y_0):
    if func_diff_x.subs({x_sym: x_0}) > 1 or func_diff_y.subs({y_sym: y_0}) > 1:
        return False
    elif phi_diff_x.subs({x_sym: x_0}) > 1 or phi_diff_y.subs({y_sym: y_0}) > 1:
        return False
    
    return True

def eps(e):
    return int(len(str(e).split('.')[1])) + 1

def simple_iter(x_0, y_0, e):
    x, y = x_0, y_0
    i = 0
    dec_point = eps(e)
    x_new = func.subs({y_sym: y})
    y_new = phi.subs({x_sym: x})
    delta_x = sp.Abs(x - x_new)
    delta_y = sp.Abs(y - y_new)
    print(f"Шаг {i + 1}:")
    print(f"x{i} = {x:.{dec_point}f}")
    print(f"x{i+1} = {x_new:.{dec_point}f}")
    print(f"y{i} = {y:.{dec_point}f}")
    print(f"y{i+1} = {y_new:.{dec_point}f}")
    print(f"|x{i} - x{i+1}| =  {delta_x:.{dec_point}f}")
    print(f"|y{i} - y{i+1}| = {delta_y:.{dec_point}f}\n")

    while delta_x > e or delta_y > e:
        i += 1
        x, y = x_new, y_new
        x_new = func.subs({y_sym: y})
        y_new = phi.subs({x_sym: x})
        delta_x = sp.Abs(x - x_new)
        delta_y = sp.Abs(y - y_new)
        print(f"Шаг {i + 1}:")
        print()
        print(f"x{i} = {x:.{dec_point}f}")
        print(f"x{i+1} = {x_new:.{dec_point}f}")
        print(f"y{i} = {y:.{dec_point}f}")
        print(f"y{i+1} = {y_new:.{dec_point}f}")
        print(f"|x{i} - x{i+1}| =  {delta_x:.{dec_point}f}")
        print(f"|y{i} - y{i+1}| = {delta_y:.{dec_point}f}\n")

    print(f"Условия выполнены: |x{i} - x{i+1}| = {delta_x:.{dec_point}f} < {epsilon:.{dec_point}f}, |y{i} - y{i+1}| = {delta_y:.{dec_point}f} < {epsilon:.{dec_point}f}")
    return x_new, y_new

def solution_graph(a = -2, b = 2):
    x_vals = np.linspace(a, b, 400)
    y_vals = np.linspace(a, b, 400)

    f_1 = sp.cos(y_sym + 0.5) + x_sym - 0.8
    f_2 = sp.sin(x_sym) - 2*y_sym - 1.6

    f_1_np = sp.lambdify((x_sym, y_sym), f_1, 'numpy')
    f_2_np = sp.lambdify((x_sym, y_sym), f_2, 'numpy')

    x_grid, y_grid = np.meshgrid(x_vals, y_vals)
    res_1 = f_1_np(x_grid, y_grid)
    res_2 = f_2_np(x_grid, y_grid)


    plt.axhline(0, color='black', linewidth=1.3, linestyle='-')  
    plt.axvline(0, color='black', linewidth=1.3, linestyle='-')
    plt.contour(x_grid, y_grid, res_1, levels = [0], color = 'purple')
    plt.contour(x_grid, y_grid, res_2, levels=[0], colors='red')
    plt.scatter([x_res], [y_res], color='black', zorder=3, label=f'Решение ({x_res:.{dec_point}f}, {y_res:.{dec_point}f})')
    plt.annotate(r'$\cos(y + 0.5) + x - 0.8 = 0$', xy=(-1.5, 0.5), color='purple', fontsize=10)
    plt.annotate(r'$\sin(x) - 2y - 1.6 = 0$', xy=(0.5, -1), color='red', fontsize=10)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.title("Метод итераций для решения системы нелинейных уравнений")
    plt.show()
    
zero_approximation_graph()

x_0 = float(input("Введите начальное приближение x: "))
y_0 = float(input("Введите начальное приближение y: "))
epsilon = float(input("Введите точность epsilon: "))
dec_point = eps(epsilon)

if convergence_conditions(x_0, y_0):
    print(f"Условия сходимости выполнены: f'(x) = {func_diff_x.subs({x_sym: x_0}):.{dec_point}f} < 1", end='; ')
    print(f"f'(y) = {func_diff_y.subs({y_sym: y_0}):.{dec_point}f} < 1", end='; ')
    print(f"phi'(x) = {phi_diff_x.subs({x_sym: x_0}):.{dec_point}f} < 1", end='; ')
    print(f"phi'(y) = {phi_diff_y.subs({y_sym: y_0}):.{dec_point}f} < 1", end=' ')
    
    x_res, y_res = simple_iter(x_0, y_0, epsilon)

    print(f"Решение: x = {x_res:.{dec_point}f}, y = {y_res:.{dec_point}f}")

    solution_graph()
else:
    print('Не выполнены условия сходимости')