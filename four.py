

import os
import numpy as np
import matplotlib.pyplot as plt

OUT_DIR = "pr4_output"
os.makedirs(OUT_DIR, exist_ok=True)


def f1(x1, x2):
    return 2*x1 - 6*x2


def f2(x1, x2):
    return 3*x1 + x2


def feasible(x1, x2, extra=None):
    eps = 1e-9
    ok = (
        x1 + x2 <= 4 + eps and
        x1 + x2 >= 1 - eps and
        x1 - 2*x2 <= 1 + eps and
        x1 >= -eps and
        x2 >= -eps
    )
    if extra is not None:
        ok = ok and extra(x1, x2)
    return ok


def polygon_vertices(extra=None):
    # Ограничивающие прямые:
    # x1+x2=4, x1+x2=1, x1-2x2=1, x1=0, x2=0
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
            if feasible(x, y, extra):
                pts.append((round(x, 10), round(y, 10)))
    pts = sorted(set(pts))
    if not pts:
        return []
    cx = sum(p[0] for p in pts) / len(pts)
    cy = sum(p[1] for p in pts) / len(pts)
    pts.sort(key=lambda p: np.arctan2(p[1] - cy, p[0] - cx))
    return pts


def solve_linear_objective(obj, sense="min", extra=None):
    pts = polygon_vertices(extra)
    if not pts:
        return None
    vals = [(obj(x, y), x, y) for x, y in pts]
    if sense == "min":
        return min(vals)
    return max(vals)


def pareto_points(grid_step=0.02):
    pts = []
    xs = np.arange(0, 4.0001, grid_step)
    ys = np.arange(0, 4.0001, grid_step)
    for x in xs:
        for y in ys:
            if feasible(x, y):
                pts.append((x, y, f1(x, y), f2(x, y)))
    pareto = []
    for p in pts:
        dominated = False
        for q in pts:
            # минимизация двух критериев
            if (q[2] <= p[2] + 1e-9 and q[3] <= p[3] + 1e-9 and
                (q[2] < p[2] - 1e-9 or q[3] < p[3] - 1e-9)):
                dominated = True
                break
        if not dominated:
            pareto.append(p)
    return pareto


def plot_region(points, optimal_points, title, path):
    plt.figure(figsize=(8, 7))

    poly = polygon_vertices()
    if poly:
        xs = [p[0] for p in poly] + [poly[0][0]]
        ys = [p[1] for p in poly] + [poly[0][1]]
        plt.fill(xs, ys, alpha=0.2, label="ОДР")
        plt.plot(xs, ys, marker="o")

    # Ограничивающие прямые
    x = np.linspace(0, 4.5, 300)
    plt.plot(x, 4 - x, label="x1+x2=4")
    plt.plot(x, 1 - x, label="x1+x2=1")
    plt.plot(x, (x - 1) / 2, label="x1-2x2=1")

    for label, opt in optimal_points.items():
        if opt is None:
            continue
        val, x1, x2 = opt
        plt.scatter([x1], [x2], s=90, label=f"{label}: ({x1:.2f}, {x2:.2f})")

    plt.xlim(0, 4.5)
    plt.ylim(0, 4.5)
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()


def main():
    print("Практическая работа №4. Вариант 32")
    print("Многокритериальное управление. Метод главного критерия.")

    print("\nИсходные данные:")
    print("f1 = 2x1 - 6x2 -> min")
    print("f2 = 3x1 + x2 -> min")
    print("Ограничения:")
    print("x1 + x2 <= 4")
    print("x1 + x2 >= 1")
    print("x1 - 2x2 <= 1")
    print("x1, x2 >= 0")

    vertices = polygon_vertices()
    print("\nВершины области допустимых решений:")
    for p in vertices:
        print(f"x=({p[0]:.4f}, {p[1]:.4f}), f1={f1(*p):.4f}, f2={f2(*p):.4f}")

    opt_f1 = solve_linear_objective(f1, "min")
    opt_f2 = solve_linear_objective(f2, "min")

    print("\nОтдельная оптимизация критериев:")
    print(f"min f1: f1={opt_f1[0]:.4f}, x=({opt_f1[1]:.4f}, {opt_f1[2]:.4f}), f2={f2(opt_f1[1], opt_f1[2]):.4f}")
    print(f"min f2: f2={opt_f2[0]:.4f}, x=({opt_f2[1]:.4f}, {opt_f2[2]:.4f}), f1={f1(opt_f2[1], opt_f2[2]):.4f}")

    # Метод главного критерия.
    # Для осмысленной демонстрации берём пороги, равные значениям второстепенного критерия
    # в компромиссной центральной точке x=(1,0), которая лежит на границе x1+x2=1 и x1-2x2=1.
    # Эти пороги пользователь может изменить.
    t2 = 3.0  # ограничение для f2 при главном f1: f2 <= t2
    t1 = 2.0  # ограничение для f1 при главном f2: f1 <= t1

    print("\nМетод главного критерия:")
    print(f"1) Главный f1, дополнительное ограничение f2 <= {t2}")
    mgk1 = solve_linear_objective(f1, "min", extra=lambda x, y: f2(x, y) <= t2 + 1e-9)
    print(f"Решение: f1={mgk1[0]:.4f}, x=({mgk1[1]:.4f}, {mgk1[2]:.4f}), f2={f2(mgk1[1], mgk1[2]):.4f}")

    print(f"2) Главный f2, дополнительное ограничение f1 <= {t1}")
    mgk2 = solve_linear_objective(f2, "min", extra=lambda x, y: f1(x, y) <= t1 + 1e-9)
    print(f"Решение: f2={mgk2[0]:.4f}, x=({mgk2[1]:.4f}, {mgk2[2]:.4f}), f1={f1(mgk2[1], mgk2[2]):.4f}")

    pareto = pareto_points(grid_step=0.02)
    print(f"\nПриближённо найдено точек Парето на сетке: {len(pareto)}")

    path1 = os.path.join(OUT_DIR, "pr4_feasible_region_mgk.png")
    plot_region(
        vertices,
        {
            "min f1": opt_f1,
            "min f2": opt_f2,
            "МГК f1": mgk1,
            "МГК f2": mgk2,
        },
        "ПР4. ОДР и решения методом главного критерия",
        path1,
    )

    plt.figure(figsize=(8, 7))
    all_pts = []
    for x1 in np.linspace(0, 4, 150):
        for x2 in np.linspace(0, 4, 150):
            if feasible(x1, x2):
                all_pts.append((f1(x1, x2), f2(x1, x2)))
    plt.scatter([p[0] for p in all_pts], [p[1] for p in all_pts], s=5, label="критериальное множество")
    plt.scatter([p[2] for p in pareto], [p[3] for p in pareto], s=10, label="Парето-сетка")
    plt.xlabel("f1")
    plt.ylabel("f2")
    plt.title("ПР4. Критериальное пространство")
    plt.grid(True)
    plt.legend()
    path2 = os.path.join(OUT_DIR, "pr4_criteria_space.png")
    plt.savefig(path2, dpi=160, bbox_inches="tight")
    plt.close()

    print("\nГрафики сохранены:")
    print(path1)
    print(path2)

    print("\nВывод:")
    print("Отдельные минимумы критериев не совпадают, поэтому задача является многокритериальной.")
    print("Метод главного критерия превращает её в однокритериальную за счёт ограничения на второй критерий.")
    print("Пороговые значения t1 и t2 можно менять в коде для анализа чувствительности компромиссного решения.")


if __name__ == "__main__":
    main()
