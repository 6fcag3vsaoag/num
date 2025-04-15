import numpy as np
def gues (A, b, tolerance = 1e-6, max_iterations = 1000) :
    n = len(b)
    x = np.zeros(n)
    for iteration in range(max_iterations):
        x_new = np.copy(x)
        for i in range(n):
            sum1 = sum(A[i][j] * x_new[j] for j in range(i))
            sum2 = sum(A[i][j] * x[j] for j in range(i + 1, n))
            x_new[i] = (b[i] - sum1 - sum2) / A[i][i]
        if np.linalg.norm(x_new - x, ord=np.inf) < tolerance:
            print(f"Решение найдено за {i+1} итераций")
            return x_new
        x = x_new
    print("Достигнуто мксимальное количество итераций")
    return x
n = int(input("Введите размерность матрицы:"))
A = np.zeros((n,n))
print("Введите элменты матрицы по строчно А:")
for i in range(n):
    row = input(f"Строка {i+1}:").split()
    A[i]=list(map(float, row))
b = np.zeros(n)
print("Введите элменты матрицы b:")
for i in range(n):
    b[i]= float(input(f"b[{i+1}]:"))
solution = gues(A, b)
print("Решение СЛАУ")
for i, val in enumerate(solution, start=1):
    print(f"x[{i}]={val:.6f}")
# 0.61 0.71 -0.05
# -1.03 -2.05 0.87 
# 2.5 -3.12 -5.03

# 0.44 -1.16 -7.5 