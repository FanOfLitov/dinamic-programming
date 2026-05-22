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

X = np.arange(0, C + delta, delta)

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
print(f"X = {list(X)}")

for i, row in enumerate(F_matrix, start=1):
    print(f"E{i}(x) = {row}")

# =========================================================
# ДИНАМИЧЕСКОЕ ПРОГРАММИРОВАНИЕ
# =========================================================

g = np.zeros((n + 1, len(X)))
choice = np.zeros((n + 1, len(X)), dtype=int)

for i in range(1, n + 1):

    for j, x in enumerate(X):

        best_value = -1
        best_x = 0

        for k, xi in enumerate(X):

            if xi <= x:

                remaining_index = int((x - xi) // delta)

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
print(f"Оптимальное распределение x* = {x_star}")

print("\nПроверка распределения:")
print(f"Сумма финансирования = {sum(x_star)}")
print(f"Ограничение C = {C}")

# =========================================================
# ТАБЛИЦА БЕЛЛМАНА
# =========================================================

print("\n" + "=" * 100)
print("ТАБЛИЦА ФУНКЦИЙ БЕЛЛМАНА")
print("=" * 100)

header = "g/x".rjust(8)

for x in X:
    header += f"{int(x):>8}"

print(header)
print("-" * len(header))

for i in range(1, n + 1):

    line = f"g{i}".rjust(8)

    for value in g[i]:
        line += f"{int(value):>8}"

    print(line)

# =========================================================
# ТАБЛИЦА ПОКАЗАТЕЛЕЙ
# =========================================================

with np.errstate(divide="ignore", invalid="ignore"):
    L_vals = F_star * (C / X)

L_vals[0] = 0

L1 = L_vals[1]

v = np.zeros_like(L_vals)
eta = np.zeros_like(L_vals)

mask = L_vals != 0

v[mask] = L_vals[mask] / L1
eta[mask] = L1 / L_vals[mask]

delta_F = np.diff(F_star, prepend=0)

print("\n" + "=" * 100)
print("ПОКАЗАТЕЛИ ДИНАМИКИ ПРИРОСТА ЭФФЕКТА")
print("=" * 100)

print(
    f"{'x':>8} | {'F*':>8} | {'ΔF':>8} | {'L':>12} | "
    f"{'v':>10} | {'η':>10}"
)
print("-" * 100)

for i in range(1, len(X)):

    print(
        f"{int(X[i]):>8} | "
        f"{int(F_star[i]):>8} | "
        f"{int(delta_F[i]):>8} | "
        f"{L_vals[i]:>12.2f} | "
        f"{v[i]:>10.4f} | "
        f"{eta[i]:>10.4f}"
    )

Eff = F_max / C

print(f"\nЭффективность реализации мероприятий Eff = F*/C = {F_max}/{C} = {Eff:.4f}")

# =========================================================
# РЕГРЕССИОННЫЙ ПОЛИНОМ
# =========================================================

coeffs_x = np.polyfit(X, F_star, 2)
a_x, b_x, c_x = coeffs_x
poly_x = np.poly1d(coeffs_x)

J = np.arange(len(X))
coeffs_j = np.polyfit(J, F_star, 2)
a_j, b_j, c_j = coeffs_j
poly_j = np.poly1d(coeffs_j)

print("\n" + "=" * 100)
print("РЕГРЕССИОННЫЙ ПОЛИНОМ")
print("=" * 100)

print("\nПолином по объёму финансирования x:")
print(f"F(x) = {a_x:.8f}x² + {b_x:.8f}x + {c_x:.8f}")

print("\nТот же полином по номеру шага j = x / Δ:")
print(f"F(j) = {a_j:.8f}j² + {b_j:.8f}j + {c_j:.8f}")

print(
    "\nПояснение: коэффициент при x² выглядит маленьким, "
    "потому что x измеряется крупными значениями финансирования."
)

print(
    "В шкале шагов j коэффициент становится нагляднее."
)

if abs(a_x) < 0.01:
    print(
        "Коэффициент при x² малый, поэтому затухание прироста эффекта происходит медленно."
    )
else:
    print(
        "Коэффициент при x² значительный, поэтому затухание выражено сильнее."
    )

# =========================================================
# ЛИНЕЙНЫЙ ПРЕДЕЛ И РАЗНОСТЬ
# =========================================================

linear_k = F_max / C
linear_poly = np.poly1d([linear_k, 0])

diff_poly = poly_x - linear_poly

print("\n" + "=" * 100)
print("ЛИНЕЙНЫЙ ПРЕДЕЛ И РАЗНОСТЬ")
print("=" * 100)

print(f"L(x) = {linear_k:.8f}x")

print("\nD(x) = F(x) - L(x)")
print(
    f"D(x) = {diff_poly.coefficients[0]:.8f}x² + "
    f"{diff_poly.coefficients[1]:.8f}x + "
    f"{diff_poly.coefficients[2]:.8f}"
)

# =========================================================
# ТОЧКА ПЕРЕСЕЧЕНИЯ
# =========================================================

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
    F_intersect = poly_x(x_intersect)
    j_intersect = x_intersect / delta

    print(f"x = {x_intersect:.2f}")
    print(f"j = x / Δ = {j_intersect:.2f}")
    print(f"F(x) = {F_intersect:.2f}")

    print("\nИнтерпретация:")
    print(
        f"Предельная точка эффективности находится около x={x_intersect:.2f}, "
        f"то есть около шага j={j_intersect:.2f}."
    )

    print(
        "До этой точки увеличение финансирования ещё можно считать эффективным."
    )

    print(
        "После этой точки прирост эффекта становится недостаточным относительно линейного предела."
    )

else:

    x_intersect = None
    F_intersect = None
    print("Действительных положительных точек пересечения не найдено.")

# =========================================================
# ГРАФИКИ
# =========================================================

X_ext = np.linspace(0, 260, 400)
F_ext = poly_x(X_ext)
L_ext = linear_k * X_ext
D_ext = F_ext - L_ext

# График 1
plt.figure(figsize=(12, 7))

for i in range(n):
    plt.plot(
        X,
        F_matrix[i],
        marker="o",
        linestyle="--",
        alpha=0.6,
        label=f"E{i + 1}(x)"
    )

plt.plot(
    X,
    F_star,
    marker="s",
    linewidth=3,
    label="F*(x)"
)

plt.title("Функции эффекта мероприятий и результат Беллмана")
plt.xlabel("Финансирование x")
plt.ylabel("Эффект F")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# График 2
plt.figure(figsize=(12, 7))

plt.plot(
    X,
    F_star,
    "bo",
    markersize=7,
    label="Точки F*(x)"
)

plt.plot(
    X_ext,
    F_ext,
    linewidth=3,
    label="Регрессионный полином F(x)"
)

plt.plot(
    X_ext,
    L_ext,
    linestyle="--",
    linewidth=3,
    label="Линейный предел L(x)"
)

if x_intersect is not None:
    plt.plot(
        x_intersect,
        F_intersect,
        "ro",
        markersize=10,
        label=f"Пересечение ≈ {x_intersect:.2f}"
    )

plt.title("Наложение регрессии и линейного предела")
plt.xlabel("Финансирование x")
plt.ylabel("Эффект F")
plt.xlim(0, 260)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# График 3
plt.figure(figsize=(12, 7))

plt.plot(
    X_ext,
    D_ext,
    linewidth=3,
    label="D(x) = F(x) - L(x)"
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

plt.title("Разность между регрессией и линейным пределом")
plt.xlabel("Финансирование x")
plt.ylabel("D(x)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# =========================================================
# ИНДИВИДУАЛЬНЫЙ ВЫВОД
# =========================================================

print("\n" + "=" * 100)
print("ИНДИВИДУАЛЬНЫЙ ВЫВОД")
print("=" * 100)

print(
    "Оптимальное распределение финансирования для варианта 32 равно "
    f"x* = {x_star}, максимальный эффект F* = {int(F_max)}."
)

print(
    "Регрессионный полином имеет отрицательный коэффициент при x², "
    "поэтому кривая эффекта является вогнутой."
)

print(
    "Коэффициент при x² малый, значит затухание прироста эффекта идёт медленно."
)

print(
    "Это подтверждается таблицей ΔF: прирост эффекта снижается не резко, "
    "а постепенно."
)

print(
    "Линейный предел L(x) показывает условную границу эффективности затрат."
)

print(
    "Разность D(x) = F(x) - L(x) позволяет определить предельную точку, "
    "после которой дальнейшее увеличение финансирования становится менее выгодным."
)

if x_intersect is not None:
    print(
        f"Для моего варианта точка пересечения находится около x={x_intersect:.2f}, "
        f"то есть около шага j={j_intersect:.2f}."
    )

print(
    "До этой точки финансирование можно считать эффективным, "
    "а после неё дальнейший рост затрат даёт недостаточный прирост эффекта."
)