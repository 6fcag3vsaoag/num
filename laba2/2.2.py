import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
import math

def method(a, b, func, diff_1, diff_2):

    if float(func.subs({x_sym: a})) * float(func.subs({x_sym: b})) >= 0:
        return False
    
    values = np.linspace(a, b, 1000)
    diff_1_vals = [diff_1.subs({x_sym : x}) for x in values]
    diff_2_vals = [diff_2.subs({x_sym: x}) for x in values]

    if not (all(diff_1_vals[i] <= diff_1_vals[i+1] for i in range(len(diff_1_vals)-1)) or all(diff_1_vals[i] <= diff_1_vals[i+1] for i in range(len(diff_1_vals)-1))):
        return False
    
    if not (all(diff_2_vals[i] <= diff_2_vals[i+1] for i in range(len(diff_2_vals)-1)) or all(diff_2_vals[i] <= diff_2_vals[i+1] for i in range(len(diff_2_vals)-1))):
        return False

    return True
def graph_1(a = 0.1, b = 2):
    x_vals = np.linspace(a, b, 400)
    y_vals_1_ = x_vals ** 3 - np.cos(x_vals + 0.5)
    #y_vals_2_ = np.cos(x_vals + 0.5)

    plt.plot(x_vals, y_vals_1_, label = r'$y = x^3$-cos(x + 0.5)', color = 'r')
    plt.axhline(0, color='black', linewidth=1.3, linestyle='-')  
    plt.axvline(0, color='black', linewidth=1.3, linestyle='-')
    #plt.plot(x_vals, y_vals_2_, label = r'$y = cos(x + 0.5)$', color = 'k')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.title("Графики функций $x^3 - cos(x + 0.5)$")
    plt.show()
def eps(e):
    e = float(e)  
    decimal_places = abs(math.floor(math.log10(e))) + 1 
    return decimal_places
def iterations(a, b, e, func, f_prime):
    a_0 = b
    b_0 = a
    a_1 = a_0
    b_1 = b_0
    i = 0
    print(f"Итерация {i}: a_{i} = {a_0:.{eps(e)}f}, b_{i} = {b_0:.{eps(e)}f}")
    print(f"|a_{i} - b_{i}| = {math.fabs(a_0 - b_0):.{eps(e)}f} >= {e}")
    print(f"x_{i} ≈ {(a_0 + b_0) / 2:.{eps(e)}f}")

    while math.fabs(a_1 - b_1) >= e:
        i += 1
        a_1 = a_0 - (func.subs({x_sym: a_0})/f_prime.subs({x_sym: a_0}))
        b_1 = b_0 - (func.subs({x_sym: b_0}) * (a_0 - b_0))/(func.subs({x_sym: a_0}) - func.subs({x_sym: b_0}))

        a_0, b_0 = a_1, b_1

        print(f"Итерация {i}: a_{i} = {a_1:.{eps(e)}f}, b_{i} = {b_1:.{eps(e)}f}")
        print(f"|a_{i} - b_{i}| = {math.fabs(a_1 - b_1):.{eps(e)}f} {'<=' if math.fabs(a_1 - b_1) < e else '>='} {e}")
        print(f"x_{i} ≈ {(a_1 + b_1) / 2:.{eps(e)}f}")
        print("----------------------")

    res = (a_1 + b_1)/2
    print(f"Приближенное решение: x ≈ {res:.{eps(e)}f} (с точностью {e})")
    #sol_graph(a, b, a_1, b_1)
def sol_graph(a, b, x_res, y_res):
    x_vals = np.linspace(a, b, 400)
    y_vals_1_ = x_vals ** 3
    y_vals_2_ = np.cos(x_vals + 0.5)
    y_res = x_res**3
    plt.plot(x_vals, y_vals_1_, label = r'$y = x^3$', color = 'r')
    plt.plot(x_vals, y_vals_2_, label = r'$y = cos(x + 0.5)$', color = 'k')
    plt.scatter([x_res], [y_res], color='black', zorder=3, label=f'Решение ({x_res:.{dec_point}f}, {y_res:.{dec_point}f})')

    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.title("График решения")
    plt.show()

x_sym = sp.Symbol('x')
func = sp.Pow(x_sym, 3) - sp.cos(x_sym + 0.5)
func_diff_1 = sp.diff(func, x_sym)
func_diff_2 = sp.diff(func_diff_1, x_sym)

graph_1()

a = float(input("Введите начало отрезка: "))
b = float(input("Введите конец отрезка: "))
epsilon = float(input("Введите точность: "))
dec_point = eps(epsilon)


if method(a, b, func, func_diff_1, func_diff_2):
    iterations(a, b, epsilon, func, func_diff_1)
else:
    print("Метод не сходится на данном отрезке")