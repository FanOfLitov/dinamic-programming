

import os
import numpy as np
import matplotlib.pyplot as plt

OUT_DIR = "pr5_output"
os.makedirs(OUT_DIR, exist_ok=True)


def f1(x1, x2):
    return 2*x1 - 6*x2


def f2(x1, x2):
    return 3*x1 + x2


def feasible(x1, x2):
    eps = 1e-9
    return (
        x1 + x2 <= 4 + eps and
        x1 + x2 >= 1 - eps and
        x1 - 2*x2 <= 1 + eps and
        x1 >= -eps and
        x2 >= -eps
    )


def vertices():
    lines = [
        (1, 1, 4),
        (1, 1, 1),
        (1, -2, 1),
        (1, 0, 0),
        (0, 1, 0),
    ]
    pts = []
    for i in range(len(lines)):
        for j in range(i + 1, len(lines)):
            a1, b1, c1 = lines[i]
            a2, b2, c2 = lines[j]
            det = a1*b2 - a2*b1
            if abs(det) < 1e-12:
                continue
            x = (c1*b2 - c2*b1) / det
            y = (a1*c2 - a2*c1) / det
            if feasible(x, y):
                pts.append((round(x, 10), round(y, 10)))
    pts = sorted(set(pts))
    cx = sum(x for x, _ in pts) / len(pts)
    cy = sum(y for _, y in pts) / len(pts)
    pts.sort(key=lambda p: np.arctan2(p[1] - cy, p[0] - cx))
    return pts


def grid_points(step=0.002):
    pts = []
    for x1 in np.arange(0, 4.0001, step):
        for x2 in np.arange(0, 4.0001, step):
            if feasible(x1, x2):
                pts.append((x1, x2))
    return pts


def normalize_min(value, f_min, f_max):
    """
    Нормализация критерия минимизации:
    0 = лучшее значение, 1 = худшее значение.
    """
    if abs(f_max - f_min) < 1e-12:
        return 0
    return (value - f_min) / (f_max - f_min)


def choose_best(points, score_func):
    best = None
    for x1, x2 in points:
        sc = score_func(x1, x2)
        if best is None or sc < best[0]:
            best = (sc, x1, x2, f1(x1, x2), f2(x1, x2))
    return best


def main():
    print("Практическая работа №5. Вариант 32")
    print("Многокритериальное управление: свёртки и метод идеальной точки.")

    omega1 = 0.3
    omega2 = 0.7
    p = 4
    q = 3

    print("\nИсходные данные:")
    print("f1 = 2x1 - 6x2 -> min")
    print("f2 = 3x1 + x2 -> min")
    print("Ограничения:")
    print("x1 + x2 <= 4")
    print("x1 + x2 >= 1")
    print("x1 - 2x2 <= 1")
    print("x1, x2 >= 0")
    print(f"ω1={omega1}, ω2={omega2}, p={p}, q={q}")

    pts_vertices = vertices()
    f1_values = [f1(x, y) for x, y in pts_vertices]
    f2_values = [f2(x, y) for x, y in pts_vertices]

    f1_min, f1_max = min(f1_values), max(f1_values)
    f2_min, f2_max = min(f2_values), max(f2_values)

    print("\nВершины ОДР и значения критериев:")
    for x, y in pts_vertices:
        print(f"x=({x:.4f}, {y:.4f}), f1={f1(x,y):.4f}, f2={f2(x,y):.4f}")

    print("\nДиапазоны критериев на ОДР:")
    print(f"f1_min={f1_min:.4f}, f1_max={f1_max:.4f}")
    print(f"f2_min={f2_min:.4f}, f2_max={f2_max:.4f}")

    points = grid_points(step=0.01)

    # 1. Аддитивная свёртка нормированных критериев минимизации.
    def additive_score(x1, x2):
        n1 = normalize_min(f1(x1, x2), f1_min, f1_max)
        n2 = normalize_min(f2(x1, x2), f2_min, f2_max)
        return omega1 * n1 + omega2 * n2

    add_best = choose_best(points, additive_score)

    # 2. Мультипликативная / степенная свёртка для минимизации.
    # Чтобы не было нулей, добавляем малый eps.
    def multiplicative_score(x1, x2):
        eps = 1e-9
        n1 = normalize_min(f1(x1, x2), f1_min, f1_max) + eps
        n2 = normalize_min(f2(x1, x2), f2_min, f2_max) + eps
        return (n1 ** omega1) * (n2 ** omega2)

    mult_best = choose_best(points, multiplicative_score)

    # 3. Максиминная свёртка для минимизации: минимизируем максимум нормированных отклонений.
    def minimax_score(x1, x2):
        n1 = normalize_min(f1(x1, x2), f1_min, f1_max)
        n2 = normalize_min(f2(x1, x2), f2_min, f2_max)
        return max(n1, n2)

    minimax_best = choose_best(points, minimax_score)

    # 4. Метод идеальной точки.
    # Идеальная точка в пространстве критериев: (f1_min, f2_min).
    # Используем L_p-метрику с p=4 и L_q-метрику с q=3.
    def ideal_distance_power(power):
        def score(x1, x2):
            n1 = normalize_min(f1(x1, x2), f1_min, f1_max)
            n2 = normalize_min(f2(x1, x2), f2_min, f2_max)
            return (omega1 * abs(n1) ** power + omega2 * abs(n2) ** power) ** (1 / power)
        return score

    ideal_p_best = choose_best(points, ideal_distance_power(p))
    ideal_q_best = choose_best(points, ideal_distance_power(q))

    solutions = {
        "Аддитивная свёртка": add_best,
        "Мультипликативная свёртка": mult_best,
        "Минимаксная свёртка": minimax_best,
        f"Идеальная точка L{p}": ideal_p_best,
        f"Идеальная точка L{q}": ideal_q_best,
    }

    print("\nРезультаты расчёта:")
    print(f"{'Метод':<30} | {'score':>10} | {'x1':>7} | {'x2':>7} | {'f1':>9} | {'f2':>9}")
    print("-" * 86)
    for name, sol in solutions.items():
        score, x1, x2, fv1, fv2 = sol
        print(f"{name:<30} | {score:>10.5f} | {x1:>7.3f} | {x2:>7.3f} | {fv1:>9.3f} | {fv2:>9.3f}")

    # График ОДР и найденных решений
    plt.figure(figsize=(8, 7))
    poly = pts_vertices
    xs = [pnt[0] for pnt in poly] + [poly[0][0]]
    ys = [pnt[1] for pnt in poly] + [poly[0][1]]
    plt.fill(xs, ys, alpha=0.2, label="ОДР")
    plt.plot(xs, ys, marker="o")

    for name, sol in solutions.items():
        _, x1, x2, _, _ = sol
        plt.scatter([x1], [x2], s=80, label=name)

    x = np.linspace(0, 4.5, 300)
    plt.plot(x, 4 - x, linewidth=1)
    plt.plot(x, 1 - x, linewidth=1)
    plt.plot(x, (x - 1) / 2, linewidth=1)
    plt.xlim(0, 4.5)
    plt.ylim(0, 4.5)
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title("ПР5. Решения ЗМУ разными методами")
    plt.grid(True)
    plt.legend(fontsize=8)
    path1 = os.path.join(OUT_DIR, "pr5_solutions_in_decision_space.png")
    plt.savefig(path1, dpi=160, bbox_inches="tight")
    plt.close()

    # График критериального пространства
    crit = [(f1(x, y), f2(x, y)) for x, y in points[::max(1, len(points)//3000)]]
    plt.figure(figsize=(8, 7))
    plt.scatter([a for a, b in crit], [b for a, b in crit], s=5, label="критериальное множество")
    for name, sol in solutions.items():
        _, _, _, fv1, fv2 = sol
        plt.scatter([fv1], [fv2], s=80, label=name)
    plt.scatter([f1_min], [f2_min], s=110, marker="*", label="идеальная точка")
    plt.xlabel("f1")
    plt.ylabel("f2")
    plt.title("ПР5. Решения в критериальном пространстве")
    plt.grid(True)
    plt.legend(fontsize=8)
    path2 = os.path.join(OUT_DIR, "pr5_solutions_in_criteria_space.png")
    plt.savefig(path2, dpi=160, bbox_inches="tight")
    plt.close()

    print("\nГрафики сохранены:")
    print(path1)
    print(path2)

    print("\nВывод:")
    print("Разные методы свёртки могут давать разные компромиссные решения.")
    print("При весах ω1=0.3 и ω2=0.7 второй критерий оказывает большее влияние,")
    print("поэтому решения смещаются в сторону уменьшения f2.")


if __name__ == "__main__":
    main()
