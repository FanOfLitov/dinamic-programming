# -*- coding: utf-8 -*-

import os
import math
import matplotlib.pyplot as plt

OUT_DIR = "pr1_output"
os.makedirs(OUT_DIR, exist_ok=True)


def solve_knapsack_dp(Y_values, w, c, lower=None, upper=None):
    m = len(w)

    if lower is None:
        lower = [0] * m

    if upper is None:
        upper = [None] * m

    max_y = max(Y_values)
    neg = -10**18

    dp = [[neg] * (max_y + 1) for _ in range(m + 1)]
    choice = [[None] * (max_y + 1) for _ in range(m + 1)]

    dp[0][0] = 0

    for i in range(1, m + 1):

        wi = w[i - 1]
        ci = c[i - 1]
        lo = lower[i - 1]
        up = upper[i - 1]

        for y in range(max_y + 1):

            best = neg
            best_x = None

            max_x = y // wi

            if up is not None:
                max_x = min(max_x, up)

            for xi in range(lo, max_x + 1):

                prev_y = y - xi * wi

                if dp[i - 1][prev_y] == neg:
                    continue

                value = dp[i - 1][prev_y] + xi * ci

                if value > best:
                    best = value
                    best_x = xi

            dp[i][y] = best
            choice[i][y] = best_x

    results = {}

    for Y in Y_values:

        best_y = max(
            range(Y + 1),
            key=lambda yy: dp[m][yy]
        )

        best_f = dp[m][best_y]

        x = [0] * m
        y = best_y

        for i in range(m, 0, -1):

            xi = choice[i][y]

            if xi is None:
                xi = 0

            x[i - 1] = xi
            y -= xi * w[i - 1]

        results[Y] = {
            "F": best_f,
            "x": x,
            "used_budget": best_y
        }

    scale = []

    for i in range(1, m + 1):
        scale.append(max(dp[i][:max_y + 1]))

    return results, scale


def print_table(title, results):
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)
    print(f"{'Y':>5} | {'F(Y)':>8} | {'исп. бюджет':>12} | {'x':>25}")
    print("-" * 100)

    for Y, r in results.items():
        print(f"{Y:>5} | {r['F']:>8} | {r['used_budget']:>12} | {str(r['x']):>25}")


def save_plot(filename, title, x_values, series, xlabel="Y", ylabel="F(Y)", xlim=None):
    plt.figure(figsize=(10, 6))

    for label, values in series.items():
        plt.plot(x_values, values, marker="o", label=label)

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    if xlim is not None:
        plt.xlim(*xlim)

    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    path = os.path.join(OUT_DIR, filename)
    plt.savefig(path, dpi=160)
    plt.close()

    return path


def analyze_task2_positions(base, minus, plus, Y_values):
    print("\n" + "=" * 100)
    print("АНАЛИЗ ПОЗИЦИЙ ВО ВТОРОЙ ЗАДАЧЕ")
    print("=" * 100)

    print(
        "Позиции x1, x2, x3, x4 показывают количество СрЗИ каждого типа "
        "в оптимальном плане."
    )

    print("\nСравниваются планы при b-1, исходном b и b+1.")
    print("-" * 100)
    print(f"{'Y':>5} | {'x(b-1)':>18} | {'x(b)':>18} | {'x(b+1)':>18} | {'ΔF-':>6} | {'ΔF+':>6}")
    print("-" * 100)

    total_minus = [0, 0, 0, 0]
    total_plus = [0, 0, 0, 0]

    for Y in Y_values:

        xb = base[Y]["x"]
        xm = minus[Y]["x"]
        xp = plus[Y]["x"]

        d_f_minus = minus[Y]["F"] - base[Y]["F"]
        d_f_plus = plus[Y]["F"] - base[Y]["F"]

        for i in range(4):
            total_minus[i] += xm[i] - xb[i]
            total_plus[i] += xp[i] - xb[i]

        print(
            f"{Y:>5} | {str(xm):>18} | {str(xb):>18} | "
            f"{str(xp):>18} | {d_f_minus:>6} | {d_f_plus:>6}"
        )

    print("\nСуммарное изменение по позициям:")
    print(f"При b-1: Δx = {total_minus}")
    print(f"При b+1: Δx = {total_plus}")

    print("\nВывод по позициям:")
    print(
        "Если компонент Δx отрицательный, соответствующий тип СрЗИ используется реже. "
        "Если положительный — чаще."
    )
    print(
        "Во второй задаче изменение b напрямую влияет на допустимое количество СрЗИ, "
        "поэтому меняется состав оптимального комплекса."
    )


def analyze_task3_special(base, changed, Y_values):
    print("\n" + "=" * 100)
    print("ОСОБЫЕ СЛУЧАИ ДЛЯ ТРЕТЬЕЙ ЗАДАЧИ")
    print("=" * 100)

    xs = [Y / 2 for Y in Y_values]

    visible = [
        i for i, X in enumerate(xs)
        if 59 <= X <= 68
    ]

    f_base = [base[Y]["F"] for Y in Y_values]
    f_changed = [changed[Y]["F"] for Y in Y_values]

    print("Диапазон анализа по графику: X = [59, 68].")
    print("Здесь X = Y / 2.")
    print("-" * 100)

    print(f"{'X':>8} | {'Y':>5} | {'F(a)':>8} | {'F(a+1)':>8} | особенность")
    print("-" * 100)

    common = []

    for i in visible:

        X = xs[i]
        Y = Y_values[i]
        fb = f_base[i]
        fc = f_changed[i]

        note = ""

        if fb == fc:
            note = "общая точка"
            common.append(i)

        print(f"{X:>8.2f} | {Y:>5} | {fb:>8} | {fc:>8} | {note}")

    print("\nОбщие точки:")
    if common:
        print(
            f"Общие точки идут на всём видимом диапазоне "
            f"от X={xs[common[0]]:.2f} до X={xs[common[-1]]:.2f}."
        )
    else:
        print("Общих точек нет.")

    print("\nСовмещённый участок:")
    if len(common) >= 2:
        print(
            f"Графики полностью совпадают на участке "
            f"X=[{xs[common[0]]:.2f}; {xs[common[-1]]:.2f}]."
        )
    else:
        print("Совмещённого участка нет.")

    print("\nГоризонтальные участки:")
    print(
        f"Исходный график F(a) горизонтален на X=[{xs[visible[0]]:.2f}; "
        f"{xs[visible[-1]]:.2f}], F=737."
    )
    print(
        f"График F(a+1) также горизонтален на X=[{xs[visible[0]]:.2f}; "
        f"{xs[visible[-1]]:.2f}], F=737."
    )

    print("\nКРАТКИЙ ИНДИВИДУАЛЬНЫЙ ВЫВОД:")
    print("Для варианта 32 в третьей задаче:")
    print("w = [5, 8, 7, 9]")
    print("c = [60, 70, 75, 89]")
    print("a = [2, 1, 1, 1]")
    print("b = [3, 2, 2, 3]")

    max_cost = 3 * 5 + 2 * 8 + 2 * 7 + 3 * 9
    max_effect = 3 * 60 + 2 * 70 + 2 * 75 + 3 * 89

    print(f"\nМаксимальный комплект по b: x = [3, 2, 2, 3].")
    print(f"Его стоимость: {max_cost}.")
    print(f"Его эффект: {max_effect}.")

    print(
        "\nМинимальный бюджет в диапазоне Y равен 119, "
        f"а максимальный комплект стоит только {max_cost}."
    )

    print(
        "Поэтому при любом Y из диапазона можно купить весь максимально допустимый комплект."
    )

    print(
        "После изменения a на a+1 = [3, 2, 2, 2] тот же комплект "
        "x = [3, 2, 2, 3] остаётся допустимым."
    )

    print(
        "Именно поэтому результат не меняется: F(a) = F(a+1) = 737."
    )

    print(
        "Ограничивающим фактором здесь является не бюджет Y и не нижняя граница a, "
        "а верхняя граница b."
    )

    print(
        "Горизонтальный участок означает насыщение: увеличение бюджета не увеличивает ЦФ, "
        "потому что все верхние ограничения b уже достигнуты."
    )


def main():
    print("=" * 100)
    print("ПРАКТИЧЕСКАЯ РАБОТА №1")
    print("Вариант 32")
    print("=" * 100)

    # =====================================================
    # ЗАДАЧА 1
    # =====================================================

    Y1 = list(range(18, 36 + 1))
    w1 = [6, 7, 9, 10, 12]
    c1 = [50, 55, 65, 70, 61]

    print("\nЗАДАЧА 1. Неограниченная задача")
    print(f"Y = [18, 36]")
    print(f"w = {w1}")
    print(f"c = {c1}")

    res1, scale1 = solve_knapsack_dp(Y1, w1, c1)
    res1_w1, _ = solve_knapsack_dp(Y1, [x - 1 for x in w1], c1)
    res1_w2, _ = solve_knapsack_dp(Y1, [x - 2 for x in w1], c1)

    print_table("Задача 1: исходные w", res1)
    print_table("Задача 1: w_i - 1", res1_w1)
    print_table("Задача 1: w_i - 2", res1_w2)

    print(f"\nДинамическая шкала для Ymax=36: {scale1}")

    p1 = save_plot(
        "task1.png",
        "Задача 1: изменение стоимости w",
        Y1,
        {
            "исходные w": [res1[Y]["F"] for Y in Y1],
            "w_i - 1": [res1_w1[Y]["F"] for Y in Y1],
            "w_i - 2": [res1_w2[Y]["F"] for Y in Y1],
        },
        xlabel="Y",
        ylabel="F(Y)"
    )

    # =====================================================
    # ЗАДАЧА 2
    # =====================================================

    Y2 = list(range(22, 44 + 1))
    w2 = [5, 9, 8, 7]
    c2 = [60, 89, 70, 75]
    b2 = [3, 1, 2, 1]

    print("\nЗАДАЧА 2. Ограниченная без обязательного применения")
    print(f"Y = [22, 44]")
    print(f"w = {w2}")
    print(f"c = {c2}")
    print(f"b = {b2}")

    res2, scale2 = solve_knapsack_dp(Y2, w2, c2, lower=[0, 0, 0, 0], upper=b2)
    res2_minus, _ = solve_knapsack_dp(Y2, w2, c2, lower=[0, 0, 0, 0], upper=[max(0, x - 1) for x in b2])
    res2_plus, _ = solve_knapsack_dp(Y2, w2, c2, lower=[0, 0, 0, 0], upper=[x + 1 for x in b2])

    print_table("Задача 2: исходные b", res2)
    print_table("Задача 2: b_i - 1", res2_minus)
    print_table("Задача 2: b_i + 1", res2_plus)

    print(f"\nДинамическая шкала для Ymax=44: {scale2}")

    analyze_task2_positions(res2, res2_minus, res2_plus, Y2)

    p2 = save_plot(
        "task2.png",
        "Задача 2: изменение верхних ограничений b",
        Y2,
        {
            "исходные b": [res2[Y]["F"] for Y in Y2],
            "b_i - 1": [res2_minus[Y]["F"] for Y in Y2],
            "b_i + 1": [res2_plus[Y]["F"] for Y in Y2],
        },
        xlabel="Y",
        ylabel="F(Y)"
    )

    # =====================================================
    # ЗАДАЧА 3
    # =====================================================

    Y3 = list(range(119, 138 + 1))
    w3 = [5, 8, 7, 9]
    c3 = [60, 70, 75, 89]
    a3 = [2, 1, 1, 1]
    b3 = [3, 2, 2, 3]

    print("\nЗАДАЧА 3. Ограниченная с обязательным применением")
    print(f"Y = [119, 138]")
    print(f"w = {w3}")
    print(f"c = {c3}")
    print(f"a = {a3}")
    print(f"b = {b3}")

    res3, scale3 = solve_knapsack_dp(Y3, w3, c3, lower=a3, upper=b3)
    res3_plus, _ = solve_knapsack_dp(Y3, w3, c3, lower=[x + 1 for x in a3], upper=b3)

    print_table("Задача 3: исходные a", res3)
    print_table("Задача 3: a_i + 1", res3_plus)

    print(f"\nДинамическая шкала для Ymax=138: {scale3}")

    X3 = [Y / 2 for Y in Y3]

    p3 = save_plot(
        "task3_special_range.png",
        "Задача 3: диапазон [59, 68]",
        X3,
        {
            "исходные a": [res3[Y]["F"] for Y in Y3],
            "a_i + 1": [res3_plus[Y]["F"] for Y in Y3],
        },
        xlabel="X = Y / 2",
        ylabel="F(X)",
        xlim=(59, 68)
    )

    analyze_task3_special(res3, res3_plus, Y3)

    print("\n" + "=" * 100)
    print("ГРАФИКИ СОХРАНЕНЫ")
    print("=" * 100)
    print(p1)
    print(p2)
    print(p3)

    print("\n" + "=" * 100)
    print("ОБЩИЙ ВЫВОД ПО ПРАКТИЧЕСКОЙ №1")
    print("=" * 100)

    print(
        "В первой задаче уменьшение стоимости СрЗИ расширяет множество допустимых решений, "
        "поэтому графики w-1 и w-2 расположены не ниже исходного графика."
    )

    print(
        "Во второй задаче изменение верхних ограничений b меняет максимальное допустимое "
        "количество СрЗИ каждого типа, поэтому меняется состав оптимального комплекса."
    )

    print(
        "В третьей задаче результат не меняется, потому что бюджет значительно больше стоимости "
        "максимального комплекта по b. Поэтому оптимум ограничивается верхними границами b."
    )


if __name__ == "__main__":
    main()