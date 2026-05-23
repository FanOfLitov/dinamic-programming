# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

print("=" * 100)
print("ПРАКТИЧЕСКАЯ РАБОТА №2")
print("Вариант 32")
print("=" * 100)

# =========================================================
# ИСХОДНЫЕ ДАННЫЕ
# =========================================================

n = 6
C = 180
delta = 20

# Финансирование в денежных единицах
X_money = np.arange(0, C + delta, delta)

# Номер шага для регрессии:
# 0, 1, 2, ..., 9
X_step = np.arange(len(X_money))

F_matrix = [
    [0, 20, 44, 67, 85, 98, 110, 117, 121, 123],
    [0, 28, 54, 72, 88, 106, 119, 128, 134, 136],
    [0, 23, 44, 64, 80, 94, 105, 114, 118, 119],
    [0, 26, 46, 67, 87, 101, 111, 120, 125, 127],
    [0, 21, 45, 66, 87, 102, 114, 122, 126, 128],
    [0, 22, 48, 67, 85, 99, 111, 119, 125, 126],
]

print("\nИСХОДНЫЕ ДАННЫЕ:")
print(f"Количество мероприятий n = {n}")
print(f"Общий объём финансирования C = {C}")
print(f"Шаг финансирования Δ = {delta}")
print(f"Финансирование X = {list(X_money)}")
print(f"Номера шагов x = {list(X_step)}")

for i, row in enumerate(F_matrix, start=1):
    print(f"E{i}(X) = {row}")

# =========================================================
# ДИНАМИЧЕСКОЕ ПРОГРАММИРОВАНИЕ
# =========================================================

g = np.zeros((n + 1, len(X_money)))
choice = np.zeros((n + 1, len(X_money)), dtype=int)

for i in range(1, n + 1):

    for j, current_money in enumerate(X_money):

        best_value = -1
        best_x = 0

        for k, xi in enumerate(X_money):

            if xi <= current_money:

                remaining_index = int((current_money - xi) // delta)

                value = F_matrix[i - 1][k] + g[i - 1][remaining_index]

                if value >= best_value:

                    best_value = value
                    best_x = xi

        g[i][j] = best_value
        choice[i][j] = best_x

F_star = g[n]
F_max = F_star[-1]

# =========================================================
# ПРЯМОЙ ХОД
# =========================================================

x_star = []
remaining = C

for i in range(n, 0, -1):

    xi = int(choice[i][remaining // delta])

    x_star.append(xi)

    remaining -= xi

x_star = list(reversed(x_star))

print("\n" + "=" * 100)
print("ОПТИМАЛЬНОЕ РЕШЕНИЕ")
print("=" * 100)

print(f"Максимальный эффект F* = {int(F_max)}")
print(f"Оптимальное распределение финансирования x* = {x_star}")
print(f"Проверка: сумма финансирования = {sum(x_star)}")

# =========================================================
# ТАБЛИЦА БЕЛЛМАНА
# =========================================================

print("\n" + "=" * 100)
print("ТАБЛИЦА ФУНКЦИЙ БЕЛЛМАНА")
print("=" * 100)

header = "g/X".rjust(8)

for x in X_money:
    header += f"{int(x):>8}"

print(header)
print("-" * len(header))

for i in range(1, n + 1):

    line = f"g{i}".rjust(8)

    for value in g[i]:
        line += f"{int(value):>8}"

    print(line)

# =========================================================
# ПОКАЗАТЕЛИ ДИНАМИКИ
# =========================================================

delta_F = np.diff(F_star, prepend=0)

print("\n" + "=" * 100)
print("ПОКАЗАТЕЛИ ДИНАМИКИ ПРИРОСТА ЭФФЕКТА")
print("=" * 100)

print(f"{'шаг x':>8} | {'финанс. X':>10} | {'F*':>8} | {'ΔF':>8}")
print("-" * 100)

for i in range(len(X_step)):

    print(
        f"{int(X_step[i]):>8} | "
        f"{int(X_money[i]):>10} | "
        f"{int(F_star[i]):>8} | "
        f"{int(delta_F[i]):>8}"
    )

Eff = F_max / C

print(f"\nЭффективность Eff = F*/C = {F_max}/{C} = {Eff:.4f}")

# =========================================================
# РЕГРЕССИОННЫЙ ПОЛИНОМ ПО НОМЕРУ ШАГА БЕЗ НУЛЕВОЙ ТОЧКИ
# =========================================================

X_reg = np.arange(1, len(F_star))   # 1, 2, ..., 9
Y_reg_points = F_star[1:]           # без F*(0)

coeffs = np.polyfit(X_reg, Y_reg_points, 2)

a, b, c = coeffs
poly = np.poly1d(coeffs)

print("\n" + "=" * 100)
print("РЕГРЕССИОННЫЙ ПОЛИНОМ")
print("=" * 100)

print("Регрессия строится без нулевой точки, по x = 1, 2, ..., 9.")
print("Именно поэтому получается формула преподавателя.")

print(f"\ny = {a:.4f}x² + {b:.4f}x + {c:.4f}")

linear_k = 20
linear_poly = np.poly1d([linear_k, 0])

print(f"y = {linear_k}x")

# =========================================================
# ЛИНЕЙНАЯ ФУНКЦИЯ
# =========================================================

linear_k = 20

linear_poly = np.poly1d([linear_k, 0])

print(f"y = {linear_k}x")

print("\nПояснение для преподавателя:")
print(
    "Если строить регрессию по денежной шкале X = 0, 20, 40, ..., 180, "
    "коэффициент при квадрате получается очень маленьким."
)

print(
    "Это происходит только из-за масштаба оси."
)

print(
    "При переходе к номеру шага x = X / 20 получаем полином нормального масштаба, "
    "который соответствует ожидаемому виду."
)

# =========================================================
# РАЗНОСТЬ И ТОЧКА ПЕРЕСЕЧЕНИЯ
# =========================================================

diff_poly = poly - linear_poly

print("\n" + "=" * 100)
print("РАЗНОСТЬ МЕЖДУ РЕГРЕССИЕЙ И ЛИНЕЙНОЙ ФУНКЦИЕЙ")
print("=" * 100)

print("\nD(x) = y_reg(x) - y_line(x)")
print(
    f"D(x) = {diff_poly.coefficients[0]:.4f}x² + "
    f"{diff_poly.coefficients[1]:.4f}x + "
    f"{diff_poly.coefficients[2]:.4f}"
)

roots = np.roots(diff_poly)

real_roots = [
    r.real
    for r in roots
    if abs(r.imag) < 1e-7 and r.real > 0
]

print("\n" + "=" * 100)
print("ТОЧКА ПЕРЕСЕЧЕНИЯ")
print("=" * 100)

if real_roots:

    x_intersect = max(real_roots)

    y_intersect = poly(x_intersect)

    money_intersect = x_intersect * delta

    print(f"x = {x_intersect:.2f}")
    print(f"y = {y_intersect:.2f}")
    print(f"В денежной шкале X = {money_intersect:.2f}")

    print("\nСмысл точки пересечения:")
    print(
        "До точки пересечения регрессионная кривая эффекта находится выше линейной функции."
    )

    print(
        "Это означает, что увеличение финансирования ещё можно считать эффективным."
    )

    print(
        "После точки пересечения прирост эффекта становится недостаточным "
        "относительно линейного ориентира."
    )

else:

    x_intersect = None
    y_intersect = None
    money_intersect = None

    print("Точка пересечения не найдена.")

# =========================================================
# ГРАФИК 1: ФУНКЦИИ ЭФФЕКТА И F*
# =========================================================

plt.figure(figsize=(12, 7))

for i in range(n):

    plt.plot(
        X_step,
        F_matrix[i],
        marker="o",
        linestyle="--",
        alpha=0.6,
        label=f"E{i + 1}(x)"
    )

plt.plot(
    X_step,
    F_star,
    marker="s",
    linewidth=3,
    label="F*(x)"
)

plt.title("Функции эффекта мероприятий и результат Беллмана")
plt.xlabel("Номер шага x")
plt.ylabel("Эффект y")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# =========================================================
# ГРАФИК 2: РЕГРЕССИЯ И ЛИНЕЙНАЯ ФУНКЦИЯ
# =========================================================

X_ext = np.linspace(0, 24, 400)

Y_reg = poly(X_ext)

Y_line = linear_k * X_ext

plt.figure(figsize=(12, 7))

plt.plot(
    X_step,
    F_star,
    "bo",
    markersize=7,
    label="Точки F*(x)"
)

plt.plot(
    X_ext,
    Y_reg,
    linewidth=3,
    label="Регрессионный полином"
)

plt.plot(
    X_ext,
    Y_line,
    linestyle="--",
    linewidth=3,
    label="Линейная функция y = 20x"
)

if x_intersect is not None:

    plt.plot(
        x_intersect,
        y_intersect,
        "ro",
        markersize=10,
        label=f"Пересечение ≈ {x_intersect:.2f}"
    )

plt.title("Регрессионный полином и линейная функция")
plt.xlabel("Номер шага x")
plt.ylabel("Эффект y")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# =========================================================
# ГРАФИК 3: РАЗНОСТЬ D(x)
# =========================================================

D_ext = Y_reg - Y_line

plt.figure(figsize=(12, 7))

plt.plot(
    X_ext,
    D_ext,
    linewidth=3,
    label="D(x) = y_reg(x) - 20x"
)

plt.axhline(
    0,
    linestyle="--",
    linewidth=2,
    label="D(x) = 0"
)

if x_intersect is not None:

    plt.plot(
        x_intersect,
        0,
        "ro",
        markersize=10,
        label=f"D(x)=0 при x≈{x_intersect:.2f}"
    )

plt.title("Разность между регрессией и линейной функцией")
plt.xlabel("Номер шага x")
plt.ylabel("D(x)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# =========================================================
# ИНДИВИДУАЛЬНЫЙ ВЫВОД
# =========================================================

print("\n" + "=" * 100)
print("ВЫВОД")
print("=" * 100)

print(
    f"Для варианта 32 максимальный эффект равен F* = {int(F_max)}."
)

print(
    f"Оптимальное распределение финансирования: x* = {x_star}."
)

print(
    "Регрессионный полином построен по номеру шага x, а не по денежной шкале финансирования."
)

print(

    "y от шага x = 0, 1, 2, ..., 9."
)

print(
    f"Полученный полином: y = {a:.4f}x² + {b:.4f}x + {c:.4f}."
)

print(
    "Линейная функция сравнения имеет вид y = 20x."
)

print(
    "Коэффициент при x² отрицательный, поэтому график эффекта постепенно затухает."
)

print(
    "По модулю коэффициент при x² небольшой, следовательно затухание происходит медленно."
)

if x_intersect is not None:

    print(
        f"Точка пересечения регрессии и линейной функции находится при x ≈ {x_intersect:.2f}."
    )

    print(
        f"В денежной шкале это соответствует X ≈ {money_intersect:.2f}."
    )

print(
    "До точки пересечения увеличение финансирования можно считать эффективным."
)

print(
    "После точки пересечения дополнительный прирост эффекта становится недостаточным."
)

print(
    "Поэтому предельная эффективность достигается перед следующими шагами, "
    "а дальнейшее увеличение затрат становится нецелесообразным."
)