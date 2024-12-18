import cv2
import argparse
import os
import numpy as np

# Настройка аргументов командной строки
parser = argparse.ArgumentParser(description="Обработка видео с эффектом 'ауры'.")
parser.add_argument("video_file", type=str, help="Путь к видеофайлу для обработки.")
args = parser.parse_args()

video_path = args.video_file

# Проверяем, что файл существует
if not os.path.exists(video_path):
    print(f"Ошибка: Файл {video_path} не найден!")
    exit()

# Настройка для сохранения видео
output_path = "output_video.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Ошибка: Не удалось открыть видеофайл!")
    exit()

# Параметры для записи видео
fps = 30.0
scale = 0.5
frame_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) * scale),
              int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * scale))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, frame_size)

# Функция для определения направления движения
def calculate_flow_direction(flow, threshold=2.0):
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    mask = mag > threshold
    avg_angle = np.mean(ang[mask]) if np.any(mask) else None

    if avg_angle is None:
        return "stationary"
    if 0 <= avg_angle < np.pi / 4 or 7 * np.pi / 4 <= avg_angle <= 2 * np.pi:
        return "right"
    elif np.pi / 4 <= avg_angle < 3 * np.pi / 4:
        return "down"
    elif 3 * np.pi / 4 <= avg_angle < 5 * np.pi / 4:
        return "left"
    elif 5 * np.pi / 4 <= avg_angle < 7 * np.pi / 4:
        return "up"
    return "stationary"

# Основной алгоритм
ret, prev_frame = cap.read()
prev_frame = cv2.resize(prev_frame, None, fx=scale, fy=scale)
prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

frame_skip = 2
frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    if frame_count % frame_skip != 0:
        continue

    frame = cv2.resize(frame, None, fx=scale, fy=scale)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Оптический поток
    flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 10, 2, 3, 1.1, 0)
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])

    # Создаем тепловую карту
    heatmap = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    heatmap = np.uint8(heatmap)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # Смешиваем тепловую карту с оригинальным кадром
    aura_effect = cv2.addWeighted(frame, 0.6, heatmap_color, 0.4, 0)

    # Определяем направление движения
    direction = calculate_flow_direction(flow)

    # Добавляем текст на кадр
    cv2.putText(aura_effect, f"Direction: {direction}", (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(aura_effect, f"Processing: {os.path.basename(video_path)}",
                (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    prev_gray = gray
    out.write(aura_effect)

cap.release()
out.release()
print(f"Обработанное видео сохранено как {output_path}")
