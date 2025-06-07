import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from tqdm import tqdm

def analyze_scene_change_periods(signal, fps, output_path='scene_spectrum.png', top_n=5):
    smoothed = gaussian_filter1d(signal, sigma=5)

    # yf = np.abs(fft(smoothed))
    # xf = fftfreq(len(smoothed), d=1/fps)
    yf = np.abs(fft(signal))
    xf = fftfreq(len(signal), d=1 / fps)

    pos_mask = xf > 0
    xf = xf[pos_mask]
    yf = yf[pos_mask]
    periods = 1 / xf

    # Найдём пики в спектре
    peaks, _ = find_peaks(yf)
    peak_periods = periods[peaks]
    peak_amps = yf[peaks]

    # Выберем top_n самых ярких периодов
    sorted_indices = np.argsort(peak_amps)[-top_n:][::-1]

    print(f"\nТоп-{top_n} выраженных периодов смены сцен:")
    for i in sorted_indices:
        print(f"Период: {peak_periods[i]:.2f} сек — Амплитуда: {peak_amps[i]:.2f}")

    # График
    plt.figure(figsize=(10, 5))
    plt.plot(periods, yf)
    plt.scatter(peak_periods[sorted_indices], peak_amps[sorted_indices], color='red', label='Пики')
    plt.xlabel("Период смены сцены (сек)")
    plt.ylabel("Амплитуда")
    plt.title("Спектр периодичности смен сцен")
    plt.grid(True)
    # plt.xscale("log")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"\nГрафик спектра сохранён в: {output_path}")

    # Нормализуем спектр
    yf_norm = yf / np.max(yf) if np.max(yf) != 0 else yf

    # Построение графика
    plt.figure(figsize=(12, 6))
    plt.plot(xf, yf_norm)
    plt.title("Спектр смен сцен")
    plt.xlabel("Период (секунды)")
    plt.ylabel("Нормализованная мощность")
    plt.grid(True)
    plt.savefig("scene_change_spectrum_normalized.png")
    plt.close()
    print(f"\nГрафик спектра сохранён в: scene_change_spectrum_normalized.png")

def load_frames_from_folder(folder):
    files = sorted([
        f for f in os.listdir(folder)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ])
    frames = []
    for file in files:
        path = os.path.join(folder, file)
        frame = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        frames.append(frame)
    return frames

def detect_scene_changes(frames, threshold=3):
    changes = []
    for i in range(1, len(frames)):
        diff = cv2.absdiff(frames[i], frames[i-1])
        score = np.mean(diff)
        print(f"Сигнал смены сцены на кадре {i}: {score}")
        if score > threshold:
            changes.append(i)
    return changes


def compute_scene_change_signal(frames_folder, scenapp = False):
    scene_change_signal = []
    timescenechangearr = [0]
    prev_gray = None
    frame_files = sorted([
        f for f in os.listdir(frames_folder)
        if f.lower().endswith(('.jpg', '.png'))
    ])

    for fname in tqdm(frame_files, desc="Обработка кадров"):
        frame_path = os.path.join(frames_folder, fname)
        frame = cv2.imread(frame_path)
        if frame is None:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if prev_gray is not None:
            diff = cv2.absdiff(gray, prev_gray)
            score = np.mean(diff)

            if score > 35:
                timescenechangearr.append(1)
                print(f"Сигнал смены сцены на кадре {fname}: {score}")
            else:
                timescenechangearr.append(0)
            if scenapp:
                scene_change_signal.append(score)
            # scene_change_signal.append(score)
        prev_gray = gray

    npy_path = os.path.join(frames_folder, 'time_scene_change.npy')
    np.save(npy_path, timescenechangearr)

    return np.array(scene_change_signal)

def get_fps_from_timestamps(timestamps_path):
    with open(timestamps_path) as f:
        times = [float(line.strip().split('\t')[1].split()[0]) for line in f]

    if len(times) < 2:
        raise ValueError("Недостаточно данных для расчёта FPS.")

    intervals = [t2 - t1 for t1, t2 in zip(times[:-1], times[1:])]
    avg_interval = sum(intervals) / len(intervals)
    fps = 1 / avg_interval
    return fps


def analyze_scene_change_periods_old(changes, total_frames, fps, output_path='scene_periods_spectrum.png'):
    signal = np.zeros(total_frames)
    for idx in changes:
        signal[idx] = 1

    n = len(signal)
    yf = fft(signal)
    xf = fftfreq(n, 1 / fps)

    mask = xf > 0
    xf = xf[mask]
    yf = np.abs(yf[mask])

    with np.errstate(divide='ignore'):
        periods = 1 / xf

    valid = (periods > 1) & (periods < 60)  # от 1 до 60 сек

    plt.figure(figsize=(10, 5))
    plt.plot(periods[valid], yf[valid])
    plt.title('Спектр периодичности смен сцен')
    plt.xlabel('Период (сек)')
    plt.ylabel('Мощность')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f'График сохранён в файл: {output_path}')
    # plt.show()
    plt.close()

    # Нормализуем спектр
    yf_norm = yf / np.max(yf) if np.max(yf) != 0 else yf

    # Построение графика
    plt.figure(figsize=(12, 6))
    plt.plot(xf, yf_norm)
    plt.title("Спектр смен сцен")
    plt.xlabel("Период (секунды)")
    plt.ylabel("Нормализованная мощность")
    plt.grid(True)
    plt.savefig("scene_change_spectrum_normalized.png")
    plt.close()
    print(f"\nГрафик спектра сохранён в: scene_change_spectrum_normalized.png")

    # with open('scene_changes.txt', 'w') as f:
    #     for i in changes:
    #         f.write(f"{i}\n")
    #     print("Индексы смен сцен сохранены в scene_changes.txt")


# === Основной блок ===
if __name__ == '__main__':
    frames_folder = input("Введите путь к папке с извлечёнными кадрами: ")  # Путь к извлечённым кадрам
    fps = get_fps_from_timestamps('mult/timestamps.txt')     # Частота кадров видео

    # print("Загружаем кадры...")
    # frames = load_frames_from_folder(frames_folder)
    #
    # print("Вычисляем смены сцен...")
    # scene_changes = detect_scene_changes(frames)
    #
    # print(f"Обнаружено смен: {len(scene_changes)}")
    #
    # print("Анализируем спектр смен сцен...")
    # analyze_scene_change_periods(scene_changes, len(frames), fps)
    # timeser = []
    # timesch = np.load('mult/time_scene_change.npy')
    # for i in range(len(timesch)):
    #     timeser.append(i/fps)
    #
    # plt.plot(timesch)
    # # plt.xlabel("Время (секунды)")
    #
    # plt.show()



    print("Вычисляем сигнал смены сцен...")
    # signal = compute_scene_change_signal(frames_folder)
    compute_scene_change_signal(frames_folder)

    # npy_path = os.path.join(frames_folder, 'scene_signal.npy')
    # np.save(npy_path, signal)
    # print(f"Сигнал сохранён в: {npy_path}")
    #
    # print("Выполняется спектральный анализ...")
    # analyze_scene_change_periods(signal, fps, output_path=os.path.join(frames_folder, 'scene_spectrum.png'))