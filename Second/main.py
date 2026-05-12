import numpy as np
import matplotlib.pyplot as plt

# Данные (вариант 32)
C = 180                     # общий бюджет
step = 20                   # шаг дискретизации
levels = [0, 20, 40, 60, 80, 100, 120, 140, 160, 180]
n = 6                       # количество мероприятий

# Эффекты мероприятий при финансировании (первые 0, затем 20..180)
F = [
    [0, 20, 44, 67, 85, 98, 110, 117, 121, 123],  # мер. 1
    [0, 28, 54, 72, 88, 106, 119, 128, 134, 136], # мер. 2
    [0, 23, 44, 64, 80, 94, 105, 114, 118, 119], # мер. 3
    [0, 26, 46, 67, 87, 101, 111, 120, 125, 127], # мер. 4
    [0, 21, 45, 66, 87, 102, 114, 122, 126, 128], # мер. 5
    [0, 22, 48, 67, 85, 99, 111, 119, 125, 126]  # мер. 6
]

# Динамическое программирование
dp = [[0]*len(levels) for _ in range(n+1)]
choice = [[0]*len(levels) for _ in range(n+1)]

for i in range(1, n+1):
    for idx, x in enumerate(levels):
        best = -1
        best_k = 0
        for k in range(idx+1):
            xi = levels[k]
            prev_idx = (x - xi) // step
            val = F[i-1][k] + dp[i-1][prev_idx]
            if val > best:
                best = val
                best_k = k
        dp[i][idx] = best
        choice[i][idx] = best_k

# Оптимальное решение для C=180
budget_idx = levels.index(C)
total_effect = dp[n][budget_idx]
print(f"Максимальный суммарный эффект при C={C}: {total_effect}")

# Восстановление распределения
remaining_idx = budget_idx
alloc = [0]*n
for i in range(n, 0, -1):
    k = choice[i][remaining_idx]
    alloc[i-1] = levels[k]
    remaining_idx -= k
print(f"Оптимальное распределение средств (тыс. руб.): {alloc}")

# Расчёт показателей динамики для всех уровней бюджета от 20 до 180
Y_levels = levels[1:]  # [20,40,...,180]
F_vals = [dp[n][i] for i in range(1, len(levels))]
delta = [F_vals[0]] + [F_vals[j] - F_vals[j-1] for j in range(1, len(F_vals))]
L_vals = [F_vals[j] * C / Y_levels[j] for j in range(len(Y_levels))]
L_base = L_vals[0]
v_vals = [L / L_base for L in L_vals]
eta_vals = [1/v for v in v_vals]

print("\nПоказатели динамики прироста эффекта:")
print("Y\tF*\tδF\tL\tv\tn")
for j in range(len(Y_levels)):
    print(f"{Y_levels[j]}\t{F_vals[j]}\t{delta[j]:.0f}\t{L_vals[j]:.1f}\t{v_vals[j]:.2f}\t{eta_vals[j]:.2f}")

# Графики
plt.figure(figsize=(12, 10))
plt.subplot(2,3,1)
plt.plot(Y_levels, F_vals, 'o-', color='blue')
plt.xlabel('Бюджет Y')
plt.ylabel('F*(Y)')
plt.title('Максимальный эффект')
plt.grid(True)

plt.subplot(2,3,2)
plt.bar(Y_levels, delta, width=15, color='green', alpha=0.7)
plt.xlabel('Бюджет Y')
plt.ylabel('δF')
plt.title('Прирост эффекта')
plt.grid(True)

plt.subplot(2,3,3)
plt.plot(Y_levels, L_vals, 's-', color='red')
plt.xlabel('Бюджет Y')
plt.ylabel('L(Y)')
plt.title('Линейный предел')
plt.grid(True)

plt.subplot(2,3,4)
plt.plot(Y_levels, v_vals, 'd-', color='purple')
plt.xlabel('Бюджет Y')
plt.ylabel('v')
plt.title('Скорость прироста')
plt.grid(True)

plt.subplot(2,3,5)
plt.plot(Y_levels, eta_vals, '*-', color='orange')
plt.xlabel('Бюджет Y')
plt.ylabel('η')
plt.title('Коэффициент затухания')
plt.grid(True)

plt.tight_layout()
plt.show()