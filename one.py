# -*- coding: utf-8 -*-

import os
import math
import matplotlib.pyplot as plt

OUT_DIR = "pr1_output"
os.makedirs(OUT_DIR, exist_ok=True)


def solve_knapsack_dp(Y_values, w, c, lower=None, upper=None):
    """
    Динамическое программирование для задачи выбора комплекса СрЗИ.

    Максимизируется:
        F(x) = sum(c_i * x_i)

    Ограничение:
        sum(w_i * x_i) <= Y

    lower, upper:
        нижние и верхние границы количества средств каждого типа.
        Если upper[i] is None, средство i не ограничено сверху.
    """
    m = len(w)
    if lower is None:
        lower = [0] * m
    if upper is None:
        upper = [None] * m

    maxY = max(Y_values)
    neg = -10**18

    # dp[i][y] = максимум эффекта при бюджете y и рассмотрении первых i типов
    dp = [[neg] * (maxY + 1) for _ in range(m + 1)]
    choice = [[0] * (maxY + 1) for _ in range(m + 1)]
    dp[0][0] = 0

    for i in range(1, m + 1):
        wi, ci = w[i - 1], c[i - 1]
        lo = lower[i - 1]
        up = upper[i - 1]
        for y in range(maxY + 1):
            best_val = neg
            best_x = None
            max_x_by_budget = y // wi
            if up is None:
                hi = max_x_by_budget
            else:
                hi = min(up, max_x_by_budget)
            for xi in range(lo, hi + 1):
                prev_y = y - wi * xi
                if dp[i - 1][prev_y] == neg:
                    continue
                val = dp[i - 1][prev_y] + ci * xi
                if val > best_val:
                    best_val = val
                    best_x = xi
            dp[i][y] = best_val
            choice[i][y] = best_x if best_x is not None else None

    results = {}
    for Y in Y_values:
        # В методичке условие sum(w_i*x_i) <= Y, поэтому выбираем лучший эффект
        # среди всех затрат не больше Y, а не обязательно ровно Y.
        best_y = max(range(Y + 1), key=lambda yy: dp[m][yy])
        best_val = dp[m][best_y]
        x = [0] * m
        y = best_y
        for i in range(m, 0, -1):
            xi = choice[i][y]
            if xi is None:
                xi = 0
            x[i - 1] = xi
            y -= w[i - 1] * xi
        results[Y] = {
            "F": best_val,
            "x": x,
            "used_budget": best_y,
        }

    # Динамическая шкала для максимального Y: f_i(Ymax)
    scale = []
    for i in range(1, m + 1):
        best = max(dp[i][:maxY + 1])
        scale.append(best if best > neg // 2 else None)

    return results, scale


def print_table(title, results):
    print("\n" + "=" * 90)
    print(title)
    print("=" * 90)
    print(f"{'Y':>4} | {'F(Y)':>8} | {'исп. бюджет':>11} | план x")
    print("-" * 90)
    for Y, r in results.items():
        print(f"{Y:>4} | {r['F']:>8} | {r['used_budget']:>11} | {r['x']}")


def save_plot(filename, title, Y_values, series_dict, xlim=None, ylim=None):
    plt.figure(figsize=(10, 6))
    for label, values in series_dict.items():
        plt.plot(Y_values, values, marker="o", label=label)
    plt.title(title)
    plt.xlabel("Объём финансирования Y")
    plt.ylabel("Максимальное значение ЦФ F(Y)")
    if xlim is not None:
        plt.xlim(*xlim)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.grid(True)
    plt.legend()
    path = os.path.join(OUT_DIR, filename)
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()
    return path


def analyze_positions_task2(base_results, minus_results, plus_results, Y_values):
    """
    Анализ позиций во второй задаче.

    Под позицией понимается номер типа СрЗИ в оптимальном плане:
        x = [x1, x2, x3, x4]

    Для каждого Y сравниваем:
        - исходный план;
        - план при b_i - 1;
        - план при b_i + 1.

    Также считаем суммарное изменение по каждой позиции.
    """
    print("\n" + "=" * 90)
    print("АНАЛИЗ ПОЗИЦИЙ ВО ВТОРОЙ ЗАДАЧЕ")
    print("=" * 90)
    print("Позиции: x1, x2, x3, x4 — количество СрЗИ соответствующего типа.")
    print("Сравниваются оптимальные планы при b_i-1, исходных b_i и b_i+1.")
    print("-" * 90)
    print(f"{'Y':>4} | {'x(b-1)':>16} | {'x(b)':>16} | {'x(b+1)':>16} | {'ΔF-':>5} | {'ΔF+':>5}")
    print("-" * 90)

    total_minus = [0, 0, 0, 0]
    total_plus = [0, 0, 0, 0]
    changed_minus = []
    changed_plus = []

    for Y in Y_values:
        xb = base_results[Y]["x"]
        xm = minus_results[Y]["x"]
        xp = plus_results[Y]["x"]
        dF_minus = minus_results[Y]["F"] - base_results[Y]["F"]
        dF_plus = plus_results[Y]["F"] - base_results[Y]["F"]

        dm = [xm[i] - xb[i] for i in range(4)]
        dp = [xp[i] - xb[i] for i in range(4)]

        for i in range(4):
            total_minus[i] += dm[i]
            total_plus[i] += dp[i]

        if dm != [0, 0, 0, 0] or dF_minus != 0:
            changed_minus.append((Y, dm, dF_minus))
        if dp != [0, 0, 0, 0] or dF_plus != 0:
            changed_plus.append((Y, dp, dF_plus))

        print(f"{Y:>4} | {str(xm):>16} | {str(xb):>16} | {str(xp):>16} | {dF_minus:>5} | {dF_plus:>5}")

    print("\nСуммарное изменение количества СрЗИ по позициям относительно исходного плана:")
    print(f"При b_i - 1: Δx = {total_minus}")
    print(f"При b_i + 1: Δx = {total_plus}")

    print("\nТочки, где уменьшение b_i изменило план или значение ЦФ:")
    if changed_minus:
        for Y, dx, dF in changed_minus:
            print(f"Y={Y}: Δx={dx}, ΔF={dF}")
    else:
        print("Изменений нет.")

    print("\nТочки, где увеличение b_i изменило план или значение ЦФ:")
    if changed_plus:
        for Y, dx, dF in changed_plus:
            print(f"Y={Y}: Δx={dx}, ΔF={dF}")
    else:
        print("Изменений нет.")

    print("\nКраткий вывод по позициям:")
    print("Если Δx по позиции отрицательно, соответствующий тип СрЗИ стал использоваться реже.")
    print("Если Δx положительно, соответствующий тип СрЗИ стал использоваться чаще.")
    print("ΔF показывает изменение максимального эффекта относительно исходного ограничения b.")


def main():
    print("Практическая работа №1. Вариант 32")
    print("Метод динамического программирования: выбор комплекса средств защиты информации.")

    # ----- Задача 1. Неограниченная -----
    m1 = 5
    n1 = 19
    Y1 = list(range(18, 36 + 1))
    w1 = [6, 7, 9, 10, 12]
    c1 = [50, 55, 65, 70, 61]

    print("\nИсходные данные задачи 1 (неограниченная):")
    print(f"m={m1}, n={n1}, Y=[18,36]")
    print(f"w={w1}")
    print(f"c={c1}")

    res1, scale1 = solve_knapsack_dp(Y1, w1, c1)
    res1_w_minus_1, _ = solve_knapsack_dp(Y1, [x - 1 for x in w1], c1)
    res1_w_minus_2, _ = solve_knapsack_dp(Y1, [x - 2 for x in w1], c1)

    print_table("Задача 1: исходные стоимости w", res1)
    print_table("Задача 1: изменённые стоимости w_i^(1)=w_i-1", res1_w_minus_1)
    print_table("Задача 1: изменённые стоимости w_i^(2)=w_i-2", res1_w_minus_2)
    print(f"\nДинамическая шкала для Ymax=36: {scale1}")

    p1 = save_plot(
        "pr1_task1_unbounded.png",
        "ПР1. Задача 1: влияние изменения стоимости СрЗИ",
        Y1,
        {
            "исходные w": [res1[Y]["F"] for Y in Y1],
            "w_i - 1": [res1_w_minus_1[Y]["F"] for Y in Y1],
            "w_i - 2": [res1_w_minus_2[Y]["F"] for Y in Y1],
        },
    )

    # ----- Задача 2. Ограниченная без обязательного применения -----
    n2 = 23
    Y2 = list(range(22, 44 + 1))
    w2 = [5, 9, 8, 7]
    c2 = [60, 89, 70, 75]
    b2 = [3, 1, 2, 1]

    print("\nИсходные данные задачи 2 (ограниченная без обязательного применения):")
    print(f"m=4, n={n2}, Y=[22,44]")
    print(f"w={w2}")
    print(f"c={c2}")
    print(f"b={b2}")

    res2, scale2 = solve_knapsack_dp(Y2, w2, c2, lower=[0]*4, upper=b2)
    b2_minus = [max(0, x - 1) for x in b2]
    b2_plus = [x + 1 for x in b2]
    res2_b_minus, _ = solve_knapsack_dp(Y2, w2, c2, lower=[0]*4, upper=b2_minus)
    res2_b_plus, _ = solve_knapsack_dp(Y2, w2, c2, lower=[0]*4, upper=b2_plus)

    print_table("Задача 2: исходные ограничения b", res2)
    print_table("Задача 2: ограничения b_i^(-1)=b_i-1", res2_b_minus)
    print_table("Задача 2: ограничения b_i^(+1)=b_i+1", res2_b_plus)
    print(f"\nДинамическая шкала для Ymax=44: {scale2}")

    analyze_positions_task2(res2, res2_b_minus, res2_b_plus, Y2)

    p2 = save_plot(
        "pr1_task2_bounded.png",
        "ПР1. Задача 2: влияние изменения максимального количества b",
        Y2,
        {
            "исходные b": [res2[Y]["F"] for Y in Y2],
            "b_i - 1": [res2_b_minus[Y]["F"] for Y in Y2],
            "b_i + 1": [res2_b_plus[Y]["F"] for Y in Y2],
        },
    )

    # ----- Задача 3. Ограниченная с обязательным применением -----
    n3 = 23
    Y3 = list(range(119, 138 + 1))
    w3 = [5, 8, 7, 9]
    c3 = [60, 70, 75, 89]
    a3 = [2, 1, 1, 1]
    b3 = [3, 2, 2, 3]

    print("\nИсходные данные задачи 3 (ограниченная с обязательным применением):")
    print(f"m=4, n={n3}, Y=[119,138]")
    print(f"w={w3}")
    print(f"c={c3}")
    print(f"a={a3}")
    print(f"b={b3}")

    res3, scale3 = solve_knapsack_dp(Y3, w3, c3, lower=a3, upper=b3)
    a3_plus = [x + 1 for x in a3]
    # Если a_i+1 > b_i, верхнюю границу расширять нельзя по условию,
    # поэтому такие варианты становятся невозможными.
    res3_a_plus, _ = solve_knapsack_dp(Y3, w3, c3, lower=a3_plus, upper=b3)

    print_table("Задача 3: исходные нижние границы a", res3)
    print_table("Задача 3: изменённые нижние границы a_i^(+)=a_i+1", res3_a_plus)
    print(f"\nДинамическая шкала для Ymax=138: {scale3}")

    p3 = save_plot(
        "pr1_task3_required.png",
        "ПР1. Задача 3: влияние увеличения минимального количества a",
        Y3,
        {
            "исходные a": [res3[Y]["F"] for Y in Y3],
            "a_i + 1": [res3_a_plus[Y]["F"] if res3_a_plus[Y]["F"] > -10**17 else float("nan") for Y in Y3],
        },
        # Замечание преподавателя: ограничить диапазон третьего графика.
        # Диапазон изменения: [466.67, 1050.0].
        ylim=(466.67, 1050.0),
    )

    print("\nГрафики сохранены:")
    print(p1)
    print(p2)
    print(p3)
    print("Для третьего графика установлен диапазон по оси Y: [466.67, 1050.0].")

    print("\nВывод:")
    print("1) В неограниченной задаче уменьшение стоимости СрЗИ увеличивает доступное число средств и не уменьшает ЦФ.")
    print("2) В ограниченной задаче уменьшение b сужает множество допустимых планов, увеличение b расширяет его.")
    print("3) В задаче с обязательным применением увеличение a может резко изменить график или сделать часть решений невозможной,")
    print("   потому что минимальный обязательный комплект может стать несовместимым с верхними границами b.")


if __name__ == "__main__":
    main()
