import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linprog

# ------------------------------------------------------------
# Данные варианта 32
# ------------------------------------------------------------
# Ограничения:
# x1 + x2 <= 4
# x1 + x2 >= 1   -> -x1 - x2 <= -1
# x1 - 2x2 <= 1
# x1 >= 0, x2 >= 0

A_ub = np.array([[1, 1],
                 [-1, -1],
                 [1, -2]])
b_ub = np.array([4, -1, 1])
bounds = [(0, None), (0, None)]

# Целевые функции
def f1(x):
    return 2*x[0] - 6*x[1]

def f2(x):
    return 3*x[0] + x[1]


# Вспомогательные функции для поиска минимума/максимума

def optimize_single(c, maximize=False):
    if maximize:
        c = -np.array(c)
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    if res.success:
        x = res.x
        val = res.fun
        if maximize:
            val = -val
        return x, val
    else:
        return None, None

# Найдём min и max для f1 и f2
c_f1 = [2, -6]
c_f2 = [3, 1]

x_f1_min, f1_min = optimize_single(c_f1, maximize=False)
x_f1_max, f1_max = optimize_single(c_f1, maximize=True)
x_f2_min, f2_min = optimize_single(c_f2, maximize=False)
x_f2_max, f2_max = optimize_single(c_f2, maximize=True)

print("Предварительные расчёты")
print(f"f1_min = {f1_min:.4f} в точке {x_f1_min}")
print(f"f1_max = {f1_max:.4f} в точке {x_f1_max}")
print(f"f2_min = {f2_min:.4f} в точке {x_f2_min}")
print(f"f2_max = {f2_max:.4f} в точке {x_f2_max}")


# 1.1 Метод главного критерия

t2 = (f2_max - f2_min) / 2
t1 = (f1_max - f1_min) / 2

print("\nМетод главного критерия")

# а) Главный f1, f2 <= t2
A_ub_f1 = np.vstack([A_ub, [3, 1]])
b_ub_f1 = np.append(b_ub, t2)
res_f1 = linprog(c_f1, A_ub=A_ub_f1, b_ub=b_ub_f1, bounds=bounds, method='highs')
if res_f1.success:
    x1_mgk = res_f1.x
    f1_val = res_f1.fun
    f2_val = f2(x1_mgk)
    print(f"a) Главный f1, f2 <= {t2:.4f}:")
    print(f"   x = ({x1_mgk[0]:.4f}, {x1_mgk[1]:.4f}), f1 = {f1_val:.4f}, f2 = {f2_val:.4f}")
else:
    print("   Решения нет")

# б) Главный f2, f1 <= t1
A_ub_f2 = np.vstack([A_ub, [2, -6]])
b_ub_f2 = np.append(b_ub, t1)
res_f2 = linprog(c_f2, A_ub=A_ub_f2, b_ub=b_ub_f2, bounds=bounds, method='highs')
if res_f2.success:
    x2_mgk = res_f2.x
    f1_val = f1(x2_mgk)
    f2_val = res_f2.fun
    print(f"б) Главный f2, f1 <= {t1:.4f}:")
    print(f"   x = ({x2_mgk[0]:.4f}, {x2_mgk[1]:.4f}), f1 = {f1_val:.4f}, f2 = {f2_val:.4f}")
else:
    print("   Решения нет")


# 1.2 Метод максиминной свертки

s1 = f1_max - f1_min
s2 = f2_max - f2_min
print(f"\nНормировочные коэффициенты: s1 = {s1:.4f}, s2 = {s2:.4f}")

# Переменные: [x1, x2, t]
c = [0, 0, 1]   # минимизируем t
# Исходные ограничения, расширенные нулевым столбцом для t
A_ub_ext = np.zeros((A_ub.shape[0], 3))
A_ub_ext[:, :2] = A_ub
b_ub_ext = b_ub
# Ограничения f1 - s1*t <= 0, f2 - s2*t <= 0
A_maxmin = np.array([[2, -6, -s1],
                     [3, 1, -s2]])
b_maxmin = [0, 0]
# Объединяем
A_total = np.vstack([A_ub_ext, A_maxmin])
b_total = np.concatenate([b_ub_ext, b_maxmin])
bounds_ext = [(0, None), (0, None), (None, None)]

res_maxmin = linprog(c, A_ub=A_total, b_ub=b_total, bounds=bounds_ext, method='highs')
if res_maxmin.success:
    x_opt = res_maxmin.x[:2]
    t_opt = res_maxmin.x[2]
    f1_opt = f1(x_opt)
    f2_opt = f2(x_opt)
    print("\n=== Максиминная свёртка ===")
    print(f"   x = ({x_opt[0]:.4f}, {x_opt[1]:.4f}), f1 = {f1_opt:.4f}, f2 = {f2_opt:.4f}, t = {t_opt:.4f}")
else:
    print("\nМаксиминная свёртка: решения нет")


# 1.3 Аддитивная свёртка (с перебором k и α1)

print("\nАддитивная свёртка (F = α1*f1 + α2*f2)")
best_overall = None
best_alpha1 = None
best_k = None
best_x = None
best_f1f2 = None

for k in range(1, 10):
    r = k / 10.0
    best_for_k = None
    best_alpha_k = None
    best_x_k = None
    best_f1f2_k = None
    for alpha1 in np.arange(0.01, 1.0, 0.01):
        alpha2 = 1 - alpha1
        c_sw = [alpha1*2 + alpha2*3, alpha1*(-6) + alpha2*1]
        # Доп. ограничения: f1 ≈ r * f2
        # f1 <= 1.05*r*f2  -> 2x1-6x2 - 1.05*r*(3x1+x2) <= 0
        # f1 >= 0.95*r*f2  -> -2x1+6x2 + 0.95*r*(3x1+x2) <= 0
        a1_coef = 2 - 1.05*r*3
        a2_coef = -6 - 1.05*r*1
        b1_coef = -2 + 0.95*r*3
        b2_coef = 6 + 0.95*r*1
        A_add = np.array([[a1_coef, a2_coef],
                          [b1_coef, b2_coef]])
        b_add = [0, 0]
        A_total = np.vstack([A_ub, A_add])
        b_total = np.append(b_ub, b_add)
        res_add = linprog(c_sw, A_ub=A_total, b_ub=b_total, bounds=bounds, method='highs')
        if res_add.success:
            x_add = res_add.x
            f1_val = f1(x_add)
            f2_val = f2(x_add)
            d = 0.05 * max(abs(f1_val), abs(f2_val))
            if abs(f1_val - r*f2_val) <= d:
                obj = alpha1*f1_val + alpha2*f2_val
                if best_for_k is None or obj < best_for_k:
                    best_for_k = obj
                    best_alpha_k = alpha1
                    best_x_k = x_add
                    best_f1f2_k = (f1_val, f2_val)
    if best_for_k is not None:
        print(f"k={k}, r={r:.1f}: α1={best_alpha_k:.2f}, f1={best_f1f2_k[0]:.4f}, f2={best_f1f2_k[1]:.4f}, F={best_for_k:.4f}")
        if best_overall is None or best_for_k < best_overall:
            best_overall = best_for_k
            best_alpha1 = best_alpha_k
            best_k = k
            best_x = best_x_k
            best_f1f2 = best_f1f2_k
    else:
        print(f"k={k}: нет допустимых решений")

if best_overall is not None:
    print(f"\nЛучшее решение по аддитивной свёртке: k={best_k}, α1={best_alpha1:.2f}, α2={1-best_alpha1:.2f}")
    print(f"  x = ({best_x[0]:.4f}, {best_x[1]:.4f}), f1={best_f1f2[0]:.4f}, f2={best_f1f2[1]:.4f}")
else:
    print("Нет допустимых решений для аддитивной свёртки")


# График ОДР и полученных точек

def plot_feasible_region():
    fig, ax = plt.subplots(figsize=(8, 6))
    # Точки пересечения и ОДР
    # Найдём угловые точки вручную
    points = [(4, 0), (3, 1), (0, 4), (0, 1)]
    # Отфильтруем по ограничениям
    valid = []
    for p in points:
        if (p[0]+p[1] <= 4) and (p[0]+p[1] >= 1) and (p[0]-2*p[1] <= 1) and p[0]>=0 and p[1]>=0:
            valid.append(p)
    # Сортируем для многоугольника
    center = np.mean(valid, axis=0)
    valid_sorted = sorted(valid, key=lambda p: np.arctan2(p[1]-center[1], p[0]-center[0]))
    polygon = plt.Polygon(valid_sorted, alpha=0.2, color='gray')
    ax.add_patch(polygon)
    # Линии ограничений
    x_plot = np.linspace(0, 5, 100)
    ax.plot(x_plot, 4-x_plot, 'b-', label='x1+x2=4')
    ax.plot(x_plot, 1-x_plot, 'b--', label='x1+x2=1')
    ax.plot(x_plot, (x_plot-1)/2, 'g-', label='x1-2x2=1')
    ax.set_xlim(0, 5)
    ax.set_ylim(0, 5)
    # Отметим решения
    if 'x1_mgk' in locals() and res_f1.success:
        ax.plot(x1_mgk[0], x1_mgk[1], 'ro', label='МГК (f1 главный)')
    if 'x2_mgk' in locals() and res_f2.success:
        ax.plot(x2_mgk[0], x2_mgk[1], 'rs', label='МГК (f2 главный)')
    if 'x_opt' in locals() and res_maxmin.success:
        ax.plot(x_opt[0], x_opt[1], 'g^', label='Максиминная свёртка')
    if best_x is not None:
        ax.plot(best_x[0], best_x[1], 'm*', markersize=12, label='Аддитивная свёртка (лучшая)')
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.legend()
    ax.grid(True)
    ax.set_title('ОДР и полученные решения')
    plt.show()

plot_feasible_region()