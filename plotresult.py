import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plty

# from scipy.fft import fft, fftfreq
from collections import Counter
from scipy.signal import periodogram



timeser = []
timesch = np.load('mult/time_scene_change.npy')
for i in range(len(timesch)):
    timeser.append(i/30)

plt.plot(timesch)
# plt.xlabel("Время (секунды)")

plt.show()


# Индексы, где arr[i] == 1
indices = np.where(np.array(timesch) == 1)[0]

# Добавим начало и конец массива
full_indices = [0] + indices.tolist() + [len(timesch) - 1]

# Вычислим интервалы между всеми "событиями"
intervals = [full_indices[i+1] - full_indices[i] for i in range(len(full_indices)-1)]

print("Индексы 1 с краями:", full_indices)
print("Интервалы:", intervals)

# Построим гистограмму
counter = Counter(intervals)
periods = sorted(counter)
counts = [counter[p] for p in periods]

plt.bar(periods, counts, width=0.6)
plt.xlabel('Интервал')
plt.ylabel('Частота')
plt.title('Распределение интервалов между 1 (включая края)')
plt.grid(True)
plt.tight_layout()
plt.savefig("interval_histogram_full.png")
plt.show()

# 1. Построим частотный спектр (в частотах)
freqs, power = periodogram(intervals, scaling='spectrum')

# 2. Переведем частоту в период (P = 1/f), пропустим f=0
nonzero = freqs > 0
periods = 1 / freqs[nonzero]
power = power[nonzero]

# 3. Построим график: по оси X — период, по Y — "мощность" (частота повторения)
plty.figure(figsize=(8, 4))
plty.plot(periods, power)
plty.xlabel("Период (в индексах)")
plty.ylabel("Мощность")
plty.title("Спектр периодов интервалов между 1")
plty.grid(True)
plty.tight_layout()
plty.savefig("period_spectrum.png")
plty.show()
