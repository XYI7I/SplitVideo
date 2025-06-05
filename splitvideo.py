import os
import yt_dlp
from pytube import YouTube

import cv2
import av
from PIL import Image

def download_video(url, output_path='video.mp4'):
    ydl_opts = {
        'outtmpl': output_path,
        'format': 'bestvideo+bestaudio/best',
        'merge_output_format': 'mp4'
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])


# === 1. Скачивание видео ===
def download_youtube_video(url, output_path='video.mp4'):
    yt = YouTube(url)
    stream = yt.streams.filter(file_extension='mp4', progressive=True).get_highest_resolution()
    print(f"Скачивание: {yt.title}")
    stream.download(filename=output_path)
    print(f"Скачано в файл: {output_path}")
    return output_path


# === 2. Разделение видео на кадры ===
def split_video_into_frames(video_path, frames_dir='frames'):
    os.makedirs(frames_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    frame_count = 0

    while True:
        success, frame = cap.read()
        if not success:
            break
        frame_filename = os.path.join(frames_dir, f'frame_{frame_count:05d}.jpg')
        cv2.imwrite(frame_filename, frame)
        frame_count += 1

    cap.release()
    print(f"Извлечено {frame_count} кадров в папку: {frames_dir}")

def split_video_into_frames_with_timestamps(video_path, frames_dir='frames_tstamps'):
    os.makedirs(frames_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)  # Частота кадров видео
    frame_count = 0

    timestamps = []  # Сюда будем сохранять время каждого кадра

    while True:
        success, frame = cap.read()
        if not success:
            break

        timestamp_sec = frame_count / fps  # Время кадра в секундах
        timestamps.append(timestamp_sec)

        frame_filename = os.path.join(frames_dir, f'frame_{frame_count:05d}.jpg')
        cv2.imwrite(frame_filename, frame)
        frame_count += 1

    cap.release()

    # Сохраняем времена в текстовый файл
    with open(os.path.join(frames_dir, 'timestamps.txt'), 'w') as f:
        for i, t in enumerate(timestamps):
            f.write(f'frame_{i:05d}.jpg\t{t:.3f} sec\n')

    print(f"Извлечено {frame_count} кадров. Временные метки сохранены.")

def split_video_with_pyav(video_path, frames_dir='frames_pyav'):
    os.makedirs(frames_dir, exist_ok=True)
    container = av.open(video_path)
    stream = container.streams.video[0]

    timestamps = []
    frame_count = 0

    for frame in container.decode(stream):
        img = frame.to_image()
        # frame_filename = os.path.join(frames_dir, f'frame_{frame_count:05d}.jpg')
        # img.save(frame_filename)
        frame_filename = os.path.join(frames_dir, f'frame_{frame_count:05d}.png')
        img.save(frame_filename, quality=100)

        timestamps.append(frame.pts * stream.time_base)
        frame_count += 1

    with open(os.path.join(frames_dir, 'timestamps.txt'), 'w') as f:
        for i, t in enumerate(timestamps):
            f.write(f'frame_{i:05d}.jpg\t{t:.6f} sec\n')

    print(f"Готово: {frame_count} кадров.")



# === Пример использования ===
if __name__ == '__main__':

    # url = input('Введите ссылку на видео YouTube: ')
    video_file = 'downloaded_video.mp4'

    # download_video(url, video_file)
    # download_youtube_video(url, video_file)
    # split_video_into_frames(video_file, 'frames')
    # split_video_into_frames_with_timestamps(video_file, 'frames_tstamps')
    split_video_with_pyav(video_file, 'frames_pyav')
