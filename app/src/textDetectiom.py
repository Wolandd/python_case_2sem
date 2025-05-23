"""
Файл: textDetection.py
Назначение: обнаружение текста на изображениях с помощью EasyOCR и работа с найденными областями.
"""

import easyocr
import numpy as np
import cv2
import os
from PIL import ImageDraw, Image


def get_axis_aligned_bbox(bbox):
    """
    Преобразует поворачиваемый прямоугольник (bounding box) в осевой (не повёрнутый).

    Используется для упрощения работы с координатами текстовых блоков.

    Параметры:
        bbox: список из 4 точек, заданных как [x, y]

    Возвращает:
        словарь с координатами: x_min, y_min, x_max, y_max
    """
    x_coords = [point[0] for point in bbox]
    y_coords = [point[1] for point in bbox]

    return {
        "x_min": min(x_coords),
        "y_min": min(y_coords),
        "x_max": max(x_coords),
        "y_max": max(y_coords),
    }


def save_cropped_regions(image, image_path, detections):
    """
    Вырезает найденные области текста из изображения и сохраняет их в отдельную папку.

    Полезно для последующего анализа или хранения отдельных фрагментов документа.

    Параметры:
        image: исходное изображение (массив NumPy)
        image_path: путь к исходному изображению
        detections: список результатов OCR в формате (bbox, text, prob)
    """
    dir_name = os.path.dirname(image_path)
    base_name = os.path.basename(image_path)
    file_name = os.path.splitext(base_name)[0]
    output_dir = os.path.join(dir_name, f"{file_name}_cropped")
    os.makedirs(output_dir, exist_ok=True)

    for i, (bbox, text, prob) in enumerate(detections):
        x_min, y_min, x_max, y_max = get_axis_aligned_bbox(bbox)
        x_min, y_min, x_max, y_max = int(x_min), int(y_min), int(x_max), int(y_max)
        height, width = image.shape[:2]
        x_min = max(0, x_min)
        y_min = max(0, y_min)
        x_max = min(width, x_max)
        y_max = min(height, y_max)
        if x_max <= x_min or y_max <= y_min:
            continue  # Пропускаем некорректные области
        cropped_img = image[y_min:y_max, x_min:x_max]
        output_path = os.path.join(output_dir, f"crop_{i}.png")
        cv2.imwrite(output_path, cropped_img)
    print(f"Вырезанные изображения сохранены в {output_dir}")


def detect_text(image):
    """
    Обнаруживает текст на изображении с использованием библиотеки EasyOCR.

    Поддерживает распознавание на русском и английском языках.

    Параметры:
        image: входное изображение (путь, PIL.Image или массив NumPy)

    Возвращает:
        список словарей вида:
        [
            {'text': '...', 'bbox': {'x_min': ..., ...}, 'confidence': ...},
            ...
        ]
    """
    reader = easyocr.Reader(["ru", "en"])

    if isinstance(image, str):
        image = cv2.imread(image)

    if len(image.shape) == 3 and image.shape[2] == 3:
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    results = reader.readtext(image)

    formatted_results = []
    for bbox, text, prob in results:
        if prob > 0.5:
            aligned_bbox = get_axis_aligned_bbox(bbox)

            height, width = image.shape[:2]
            aligned_bbox["x_min"] = max(0, min(width, aligned_bbox["x_min"]))
            aligned_bbox["y_min"] = max(0, min(height, aligned_bbox["y_min"]))
            aligned_bbox["x_max"] = max(0, min(width, aligned_bbox["x_max"]))
            aligned_bbox["y_max"] = max(0, min(height, aligned_bbox["y_max"]))

            formatted_results.append(
                {"text": text, "bbox": aligned_bbox, "confidence": prob}
            )

    return formatted_results


def get_bbox_coordinates(results):
    """
    Извлекает только координаты прямоугольников из результатов OCR.

    Может использоваться для дальнейшей обработки или визуализации.

    Параметры:
        results: список результатов OCR

    Возвращает:
        список словарей с координатами прямоугольников
    """
    return [result["bbox"] for result in results]


if __name__ == "__main__":
    """
    Точка входа при запуске файла напрямую.

    Загружает пример изображения, выполняет OCR и выводит координаты найденных областей.
    """
    try:
        image_path = "../examples/example.png"
        results = detect_text(Image.open(image_path))
        bboxes = get_bbox_coordinates(results)
        print("Координаты прямоугольников:", bboxes)
    except Exception as e:
        print(f"Ошибка в главной программе: {str(e)}")
