# -*- coding: utf-8 -*-


import os
import matplotlib.pyplot as plt

OUT_DIR = "pr2_output"
os.makedirs(OUT_DIR, exist_ok=True)


def solve_financing(C, effects):
    """
    effects[i][j] = эффект i-го мероприятия при финансировании x_grid[j].
    x_grid = [0, delta, 2delta, ..., C]
    """
    n = len(effects)
    steps = len(effects[0]) - 1
    delta = C // steps
    x_grid = [j * delta for j in range(steps + 1)]

    dp = [[0] * (steps + 1) for _ in range(n + 1)]
    choice = [[0] * (steps + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        for budget_step in range(steps + 1):
            best = -10**18
            best_alloc = 0
            for alloc_step in range(budget_step + 1):
                val = effects[i - 1][alloc_step] + dp[i - 1][budget_step - alloc_step]
                if val > best:
                    best = val
                    best_alloc = alloc_step
            dp[i][budget_step] = best
            choice[i][budget_step] = best_alloc

    allocations = {}
    for budget_step in range(steps + 1):
        x = [0] * n
        b = budget_step
        for i in range(n, 0, -1):
            alloc = choice[i][b]
            x[i - 1] = alloc * delta
            b -= alloc
        allocations[x_grid[budget_step]] = x

    return x_grid, dp, choice, allocations


def main():
    print("Практическая работа №2. Вариант 32")
    print("Задача оптимального финансирования мероприятий по обеспечению ИБ.")

    n = 6
    C = 180
    delta = 20
    x_values = [20, 40, 60, 80, 100, 120, 140, 160, 180]

    # Добавляем эффект при нулевом финансировании: 0.
    effects = [
        [0, 20, 44, 67, 85, 98, 110, 117, 121, 123],
        [0, 28, 54, 72, 88, 106, 119, 128, 134, 136],
        [0, 23, 44, 64, 80, 94, 105, 114, 118, 119],
        [0, 26, 46, 67, 87, 101, 111, 120, 125, 127],
        [0, 21, 45, 66, 87, 102, 114, 122, 126, 128],
        [0, 22, 48, 67, 85, 99, 111, 119, 125, 126],
    ]

    print("\nИсходные данные:")
    print(f"n={n}, C={C}, Δ={delta}")
    print(f"x={x_values}")
    for i, row in enumerate(effects, 1):
        print(f"E{i}={row[1:]}")

    x_grid, dp, choice, allocations = solve_financing(C, effects)
    F_star = dp[n][-1]
    x_star = allocations[C]

    print("\nТаблица динамического программирования:")
    header = "x".rjust(6)
    for i in range(1, n + 1):
        header += f" | g{i}(x)".rjust(10)
    print(header)
    print("-" * len(header))
    for idx, x in enumerate(x_grid):
        line = f"{x:>6}"
        for i in range(1, n + 1):
            line += f" | {dp[i][idx]:>7}"
        print(line)

    print("\nОптимальное распределение финансирования:")
    for i, xi in enumerate(x_star, 1):
        print(f"x{i} = {xi}")
    print(f"F* = {F_star}")

    # Показатели динамики прироста эффекта
    print("\nПоказатели динамики прироста эффекта:")
    print(f"{'j':>2} | {'x_j':>5} | {'F*_j':>6} | {'δF_j':>6} | {'L_j':>8} | {'v_j':>8} | {'η_j':>8}")
    print("-" * 64)

    prev = 0
    dynamics = []
    for j in range(1, len(x_grid)):
        xj = x_grid[j]
        Fj = dp[n][j]
        dF = Fj - prev
        # По формуле методички: L_j = F*_j * C / x_j
        Lj = Fj * C / xj if xj != 0 else 0
        vj = F_star / Lj if Lj != 0 else 0
        etaj = 1 / vj if vj != 0 else 0
        dynamics.append((j, xj, Fj, dF, Lj, vj, etaj))
        print(f"{j:>2} | {xj:>5} | {Fj:>6} | {dF:>6} | {Lj:>8.2f} | {vj:>8.3f} | {etaj:>8.3f}")
        prev = Fj

    Eff = F_star / C
    print(f"\nЭффективность реализации мероприятий Eff = F*/C = {F_star}/{C} = {Eff:.4f}")

    # График оптимального эффекта от бюджета
    plt.figure(figsize=(10, 6))
    plt.plot(x_grid, dp[n], marker="o", label="Оптимальный эффект g_n(x)")
    plt.title("ПР2. Оптимальный эффект от объёма финансирования")
    plt.xlabel("Объём финансирования x")
    plt.ylabel("Максимальный эффект")
    plt.grid(True)
    plt.legend()
    path1 = os.path.join(OUT_DIR, "pr2_optimal_effect.png")
    plt.savefig(path1, dpi=160, bbox_inches="tight")
    plt.close()

    # График прироста эффекта
    plt.figure(figsize=(10, 6))
    plt.plot([d[1] for d in dynamics], [d[3] for d in dynamics], marker="o", label="δF_j")
    plt.title("ПР2. Прирост эффекта при увеличении финансирования")
    plt.xlabel("Объём финансирования x")
    plt.ylabel("Прирост эффекта δF")
    plt.grid(True)
    plt.legend()
    path2 = os.path.join(OUT_DIR, "pr2_effect_growth.png")
    plt.savefig(path2, dpi=160, bbox_inches="tight")
    plt.close()

    # График коэффициента затухания
    plt.figure(figsize=(10, 6))
    plt.plot([d[1] for d in dynamics], [d[6] for d in dynamics], marker="o", label="η_j")
    plt.title("ПР2. Коэффициент затухания динамики прироста эффекта")
    plt.xlabel("Объём финансирования x")
    plt.ylabel("η")
    plt.grid(True)
    plt.legend()
    path3 = os.path.join(OUT_DIR, "pr2_decay_coefficient.png")
    plt.savefig(path3, dpi=160, bbox_inches="tight")
    plt.close()

    print("\nГрафики сохранены:")
    print(path1)
    print(path2)
    print(path3)

    print("\nВывод:")
    print("Оптимальное распределение найдено методом динамического программирования.")
    print("По мере роста финансирования предельный прирост эффекта обычно снижается,")
    print("что отражается в изменении δF и росте/изменении коэффициента затухания η.")


if __name__ == "__main__":
    main()
