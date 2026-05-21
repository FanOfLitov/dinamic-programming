import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate

def solve_mandatory(m, w, c, a, b, Y_values):
    max_Y = max(Y_values)
    INF_NEG = -10**9
    dp = np.full((m+1, max_Y+1), INF_NEG, dtype=int)
    choice = np.zeros((m+1, max_Y+1), dtype=int)
    # Базовый случай: для 0 типов и любого бюджета эффект 0
    for y in range(max_Y+1):
        dp[0][y] = 0
    for i in range(1, m+1):
        wi = w[i-1]
        ci = c[i-1]
        ai = a[i-1]
        bi = b[i-1]
        for y in range(max_Y+1):
            best_val = INF_NEG
            best_x = 0
            min_x = ai
            max_x = min(y // wi, bi)
            if min_x > max_x:
                dp[i][y] = INF_NEG
                continue
            for x in range(min_x, max_x+1):
                prev_y = y - wi*x
                if prev_y >= 0 and dp[i-1][prev_y] > INF_NEG:
                    val = ci*x + dp[i-1][prev_y]
                    if val > best_val:
                        best_val = val
                        best_x = x
            dp[i][y] = best_val
            choice[i][y] = best_x
    f_max = {}
    x_opt = {}
    for Y in Y_values:
        if dp[m][Y] <= INF_NEG//2:
            f_max[Y] = None
            x_opt[Y] = None
        else:
            y = Y
            q = [0]*m
            for i in range(m, 0, -1):
                xi = choice[i][y]
                q[i-1] = int(xi)
                y -= w[i-1] * xi
            f_max[Y] = int(dp[m][Y])
            x_opt[Y] = q
    return f_max, x_opt, dp

# Данные для варианта 32 (из таблицы 1.13)
Y3_range = list(range(59, 69))
w3 = [5, 8, 7, 9]
c3 = [60, 70, 75, 89]
a3 = [2, 1, 1, 1]
b3 = [3, 2, 2, 3]

f3_orig, x3_orig, dp3_orig = solve_mandatory(4, w3, c3, a3, b3, Y3_range)

min_budget = sum(a3[i]*w3[i] for i in range(4))
print(f"Минимальный бюджет: {min_budget}")

# Вывод таблицы
print("\nВсе точки (Y, F(Y), x) для диапазона 59..68:")
table3 = []
for y in Y3_range:
    if f3_orig[y] is not None:
        table3.append([y, f3_orig[y], x3_orig[y]])
    else:
        table3.append([y, "нет решения", None])
print(tabulate(table3, headers=["Y", "F(Y)", "x = (x1..x4)"], tablefmt="pretty"))

# График (только для существующих решений)
Y3_valid = [y for y in Y3_range if f3_orig[y] is not None]
f3_orig_valid = [f3_orig[y] for y in Y3_valid]

if Y3_valid:
    plt.figure(figsize=(10,6))
    plt.plot(Y3_valid, f3_orig_valid, 'o-', label='Исходные a')
    plt.xlabel('Бюджет Y')
    plt.ylabel('Максимальный эффект F(Y)')
    plt.title('Задача 3: зависимость эффекта от бюджета (диапазон 59..68)')
    plt.legend()
    plt.grid(True)
    plt.show()
else:
    print("Нет допустимых решений в заданном диапазоне.")