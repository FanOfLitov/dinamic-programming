# -*- coding: utf-8 -*-

import os
import matplotlib.pyplot as plt

OUT_DIR = "pr3_output"
os.makedirs(OUT_DIR, exist_ok=True)


def solve_replacement(p, t0, n, e, s, z):
    """
    Задача о замене ТСЗИ.

    На каждом году выбираем:
    - сохранить;
    - заменить.

    При равенстве эффектов выбираем сохранение, как в методичке.
    """
    max_t = len(e) - 1
    # Нужно хранить возраста до max_t. Если возраст выйдет за таблицу,
    # считаем, что дальше используются значения последнего доступного возраста.
    def val(arr, t):
        return arr[min(t, max_t)]

    g = [[0] * (max_t + n + 2) for _ in range(n + 1)]
    decision = [[""] * (max_t + n + 2) for _ in range(n + 1)]

    for i in range(1, n + 1):
        for t in range(max_t + n + 1):
            keep_effect = val(e, t) - val(z, t) + g[i - 1][t + 1]
            replace_effect = val(s, t) - p + val(e, 0) - val(z, 0) + g[i - 1][1]
            if keep_effect >= replace_effect:
                g[i][t] = keep_effect
                decision[i][t] = "сохранить"
            else:
                g[i][t] = replace_effect
                decision[i][t] = "заменить"

    # Прямой ход
    sequence = []
    ages = []
    t = t0
    for year in range(1, n + 1):
        remaining = n - year + 1
        d = decision[remaining][t]
        sequence.append(d)
        ages.append(t)
        if d == "сохранить":
            t += 1
        else:
            t = 1

    F_star = g[n][t0]
    return g, decision, sequence, ages, F_star


def main():
    print("Практическая работа №3. Вариант 32")
    print("Задача о замене средства защиты информации.")

    p = 30
    t0 = 3
    n = 8
    t_values = list(range(9))
    e = [51.0, 49.0, 47.3, 45.6, 45.1, 44.5, 43.2, 42.6, 41.8]
    s = [10.2, 10.4, 10.8, 11.3, 11.7, 12.0, 12.4, 12.9, 13.4]
    z = [12.2, 10.4, 8.8, 7.3, 6.2, 5.3, 4.5, 3.8, 3.2]

    print("\nИсходные данные:")
    print(f"p={p}, t0={t0}, n={n}")
    print(f"t={t_values}")
    print(f"e(t)={e}")
    print(f"s(t)={s}")
    print(f"z(t)={z}")

    g, decision, sequence, ages, F_star = solve_replacement(p, t0, n, e, s, z)

    print("\nТаблица обратного хода g_i(t):")
    header = "i/t".rjust(5) + "".join([f"{t:>10}" for t in t_values])
    print(header)
    print("-" * len(header))
    for i in range(1, n + 1):
        line = f"g{i}".rjust(5)
        for t in t_values:
            line += f"{g[i][t]:>10.2f}"
        print(line)

    print("\nТаблица решений для g_i(t):")
    print(header)
    print("-" * len(header))
    for i in range(1, n + 1):
        line = f"g{i}".rjust(5)
        for t in t_values:
            line += f"{decision[i][t][:4]:>10}"
        print(line)

    print("\nПрямой ход:")
    for year, (age, d) in enumerate(zip(ages, sequence), 1):
        print(f"Год {year}: возраст ТСЗИ t={age}, решение: {d}")

    print(f"\nМаксимальное значение ЦФ F* = {F_star:.2f}")

    E_avg = F_star / n
    print(f"Среднегодовой эффект E = F*/n = {F_star:.2f}/{n} = {E_avg:.4f}")

    # Расчёт эффективности по формуле методички:
    # Eff = F* / (p0 + sum(p_k) + sum(z(t_m)) - sum(s(t_k)))
    # p0 принимаем равным цене нового ТСЗИ p.
    p0 = p
    purchase_sum = 0
    service_sum = 0
    residual_sum = 0

    for age, d in zip(ages, sequence):
        if d == "заменить":
            purchase_sum += p
            residual_sum += s[min(age, len(s) - 1)]
            service_sum += z[0]
        else:
            service_sum += z[min(age, len(z) - 1)]

    total_cost = p0 + purchase_sum + service_sum - residual_sum
    Eff = F_star / total_cost

    print("\nРасчёт эффективности применения ТСЗИ:")
    print(f"p0 = {p0}")
    print(f"Сумма покупок новых ТСЗИ = {purchase_sum:.2f}")
    print(f"Сумма эксплуатационных затрат = {service_sum:.2f}")
    print(f"Сумма остаточной стоимости проданных ТСЗИ = {residual_sum:.2f}")
    print(f"Итого затрат = {total_cost:.2f}")
    print(f"Eff = F*/затраты = {F_star:.2f}/{total_cost:.2f} = {Eff:.4f}")

    years = list(range(1, n + 1))
    plt.figure(figsize=(10, 6))
    plt.plot(years, ages, marker="o", label="Возраст ТСЗИ в начале года")
    replacement_years = [year for year, d in zip(years, sequence) if d == "заменить"]
    replacement_ages = [ages[year - 1] for year in replacement_years]
    if replacement_years:
        plt.scatter(replacement_years, replacement_ages, s=80, label="Замена")
    plt.title("ПР3. Оптимальная стратегия замены ТСЗИ")
    plt.xlabel("Год планового периода")
    plt.ylabel("Возраст ТСЗИ")
    plt.xticks(years)
    plt.grid(True)
    plt.legend()
    path1 = os.path.join(OUT_DIR, "pr3_replacement_strategy.png")
    plt.savefig(path1, dpi=160, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(t_values, [e[i] - z[i] for i in t_values], marker="o", label="Эффект сохранения e(t)-z(t)")
    plt.plot(t_values, [s[i] - p + e[0] - z[0] for i in t_values], marker="o", label="Эффект замены s(t)-p+e(0)-z(0)")
    plt.title("ПР3. Сравнение эффекта сохранения и замены")
    plt.xlabel("Возраст ТСЗИ t")
    plt.ylabel("Одногодичный эффект")
    plt.grid(True)
    plt.legend()
    path2 = os.path.join(OUT_DIR, "pr3_keep_vs_replace.png")
    plt.savefig(path2, dpi=160, bbox_inches="tight")
    plt.close()

    print("\nГрафики сохранены:")
    print(path1)
    print(path2)

    print("\nВывод:")
    print("Последовательность решений показывает годы, в которых выгодно заменить ТСЗИ.")
    print("Эффективность рассчитана с учётом цены исходного средства, покупок новых, обслуживания и остаточной стоимости.")


if __name__ == "__main__":
    main()
