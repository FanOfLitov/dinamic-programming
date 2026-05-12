import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# 1. Неограниченная задача
# ------------------------------------------------------------
def solve_unbounded(m, w, c, Y_values):
    max_Y = max(Y_values)
    dp = np.zeros((m+1, max_Y+1), dtype=int)
    choice = np.zeros((m+1, max_Y+1), dtype=int)
    for i in range(1, m+1):
        wi = w[i-1]
        ci = c[i-1]
        for y in range(max_Y+1):
            max_x = y // wi
            best_val = dp[i-1][y]
            best_x = 0
            for x in range(1, max_x+1):
                val = ci * x + dp[i-1][y - wi*x]
                if val > best_val:
                    best_val = val
                    best_x = x
            dp[i][y] = best_val
            choice[i][y] = best_x
    f_max = {}
    x_opt = {}
    for Y in Y_values:
        y = Y
        q = [0]*m
        for i in range(m, 0, -1):
            xi = choice[i][y]
            q[i-1] = int(xi)
            y -= w[i-1] * xi
        f_max[Y] = int(dp[m][Y])
        x_opt[Y] = q
    return f_max, x_opt, dp

# Данные первой задачи
m1 = 5
Y1_min, Y1_max = 18, 36
Y1_range = list(range(Y1_min, Y1_max+1))
w1 = [6, 7, 9, 10, 12]
c1 = [50, 55, 65, 70, 61]

f_orig, x_orig, dp_orig = solve_unbounded(m1, w1, c1, Y1_range)
print("Задача 1 (неограниченная)")
print(f"Динамическая шкала для Y={Y1_max}: {[int(dp_orig[i][Y1_max]) for i in range(1, m1+1)]}")
print("Максимальный эффект и наборы средств для каждого Y (первые 5 значений):")
for i, y in enumerate(Y1_range[:5]):
    print(f"Y={y:2d}, F={f_orig[y]:3d}, x={x_orig[y]}")
print("...")

# Вариации стоимости
w_minus1 = [wi-1 for wi in w1]
w_minus2 = [wi-2 for wi in w1]
f_m1, _, _ = solve_unbounded(m1, w_minus1, c1, Y1_range)
f_m2, _, _ = solve_unbounded(m1, w_minus2, c1, Y1_range)

plt.figure(figsize=(10,6))
plt.plot(Y1_range, [f_orig[y] for y in Y1_range], 'o-', label='Исходные w')
plt.plot(Y1_range, [f_m1[y] for y in Y1_range], 's--', label='w-1')
plt.plot(Y1_range, [f_m2[y] for y in Y1_range], 'd-.', label='w-2')
plt.xlabel('Бюджет Y')
plt.ylabel('Максимальный эффект F(Y)')
plt.title('Задача 1: зависимость эффекта от бюджета')
plt.legend()
plt.grid(True)
plt.show()

# ------------------------------------------------------------
# 2. Ограниченная задача (без обязательного применения)
# ------------------------------------------------------------
def solve_bounded(m, w, c, b, Y_values):
    max_Y = max(Y_values)
    dp = np.zeros((m+1, max_Y+1), dtype=int)
    choice = np.zeros((m+1, max_Y+1), dtype=int)
    for i in range(1, m+1):
        wi = w[i-1]
        ci = c[i-1]
        bi = b[i-1]
        for y in range(max_Y+1):
            best_val = -1
            best_x = 0
            max_x = min(y // wi, bi)
            for x in range(0, max_x+1):
                prev_y = y - wi*x
                val = ci*x + dp[i-1][prev_y]
                if val > best_val:
                    best_val = val
                    best_x = x
            dp[i][y] = best_val
            choice[i][y] = best_x
    f_max = {}
    x_opt = {}
    for Y in Y_values:
        y = Y
        q = [0]*m
        for i in range(m, 0, -1):
            xi = choice[i][y]
            q[i-1] = int(xi)
            y -= w[i-1] * xi
        f_max[Y] = int(dp[m][Y])
        x_opt[Y] = q
    return f_max, x_opt, dp

# Данные второй задачи
Y2_min, Y2_max = 22, 44
Y2_range = list(range(Y2_min, Y2_max+1))
w2 = [5, 9, 8, 7]
c2 = [60, 89, 70, 75]
b2 = [3, 1, 2, 1]

f2_orig, x2_orig, dp2_orig = solve_bounded(4, w2, c2, b2, Y2_range)
b_minus1 = b2.copy()
b_minus1[0] = b2[0] - 1
b_plus1 = b2.copy()
b_plus1[0] = b2[0] + 1
f2_m1, _, _ = solve_bounded(4, w2, c2, b_minus1, Y2_range)
f2_p1, _, _ = solve_bounded(4, w2, c2, b_plus1, Y2_range)

print("\nЗадача 2 (ограниченная без обязательного применения)")
print(f"Исходные b = {b2}")
print(f"b1-1 = {b_minus1[0]}, b1+1 = {b_plus1[0]}")
scale2 = [int(dp2_orig[i][Y2_max]) for i in range(1, 5)]
print(f"Динамическая шкала для Y={Y2_max}: {scale2}")
print("Примеры решений (Y, F, x):")
for y in [22, 30, 44]:
    print(f"Y={y:2d}, F={f2_orig[y]:3d}, x={x2_orig[y]}")

plt.figure(figsize=(10,6))
plt.plot(Y2_range, [f2_orig[y] for y in Y2_range], 'o-', label='Исходные b')
plt.plot(Y2_range, [f2_m1[y] for y in Y2_range], 's--', label='b1-1')
plt.plot(Y2_range, [f2_p1[y] for y in Y2_range], 'd-.', label='b1+1')
plt.xlabel('Бюджет Y')
plt.ylabel('Максимальный эффект F(Y)')
plt.title('Задача 2: зависимость эффекта от бюджета (изменение b1)')
plt.legend()
plt.grid(True)
plt.show()

# ------------------------------------------------------------
# 3. Ограниченная задача с обязательным применением
# ------------------------------------------------------------
def solve_mandatory(m, w, c, a, b, Y_values):
    max_Y = max(Y_values)
    INF_NEG = -10**9
    dp = np.full((m+1, max_Y+1), INF_NEG, dtype=int)
    choice = np.zeros((m+1, max_Y+1), dtype=int)
    # База: для 0 типов можно иметь любой остаток бюджета, эффект 0
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
                # prev_y может быть любым (0..max_Y)
                val = ci*x + dp[i-1][prev_y]
                if val > best_val:
                    best_val = val
                    best_x = x
            dp[i][y] = best_val
            choice[i][y] = best_x
    f_max = {}
    x_opt = {}
    for Y in Y_values:
        if dp[m][Y] < 0:  # отрицательное значение считаем недопустимым
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

# Данные третьей задачи
Y3_min, Y3_max = 119, 138
Y3_range = list(range(Y3_min, Y3_max+1))
w3 = [5, 8, 7, 9]
c3 = [60, 70, 75, 89]
a3 = [2, 1, 1, 1]
b3 = [3, 2, 2, 3]

f3_orig, x3_orig, dp3_orig = solve_mandatory(4, w3, c3, a3, b3, Y3_range)
a_plus1 = a3.copy()
a_plus1[0] = a3[0] + 1
f3_p1, _, _ = solve_mandatory(4, w3, c3, a_plus1, b3, Y3_range)

print("\nЗадача 3 (ограниченная с обязательным применением)")
print(f"Исходные a = {a3}")
print(f"a1+1 = {a_plus1[0]}")
# Вычисляем минимальный необходимый бюджет
min_budget = sum(a3[i]*w3[i] for i in range(4))
print(f"Минимальный бюджет для исходных a: {min_budget}")
if dp3_orig[4][Y3_max] > -10**8:
    scale3 = [int(dp3_orig[i][Y3_max]) for i in range(1, 5)]
    print(f"Динамическая шкала для Y={Y3_max}: {scale3}")
else:
    print(f"Для Y={Y3_max} допустимого решения нет")
print("Примеры решений для исходных a (Y, F, x):")
for y in [119, 130, 138]:
    if f3_orig[y] is not None:
        print(f"Y={y:3d}, F={f3_orig[y]:3d}, x={x3_orig[y]}")
    else:
        print(f"Y={y:3d}, решения нет")

# графики для третьей задачи
Y3_valid = [y for y in Y3_range if f3_orig[y] is not None]
f3_orig_valid = [f3_orig[y] for y in Y3_valid]
Y3_p1_valid = [y for y in Y3_range if f3_p1[y] is not None]
f3_p1_valid = [f3_p1[y] for y in Y3_p1_valid]

plt.figure(figsize=(10,6))
plt.plot(Y3_valid, f3_orig_valid, 'o-', label='Исходные a')
plt.plot(Y3_p1_valid, f3_p1_valid, 's--', label='a1+1')
plt.xlabel('Бюджет Y')
plt.ylabel('Максимальный эффект F(Y)')
plt.title('Задача 3: зависимость эффекта от бюджета (изменение a1)')
plt.legend()
plt.grid(True)
plt.show()

print("\nВыводы")
print("1. В неограниченной задаче снижение стоимости ведёт к увеличению эффекта.")
print("2. В ограниченной задаче без обязательного применения увеличение b1 расширяет возможности выбора.")
print("3. В задаче с обязательным применением решения существуют при Y >= минимального бюджета (34).")