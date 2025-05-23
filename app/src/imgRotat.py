"""
Файл: imgRotat.py
Назначение: коррекция ориентации и наклона изображения документа
"""

import numpy as np
import cv2
from skimage import io
from skimage.transform import rotate
from skimage.color import rgb2gray
from deskew import determine_skew


def deskew(image):
    """
    Функция выравнивает изображение, определяя и исправляя угол наклона текста.

    Если изображение цветное, оно преобразуется в оттенки серого для анализа.
    Затем определяется угол наклона с помощью функции determine_skew.
    После этого изображение поворачивается на найденный угол и возвращается как результат.

    В случае ошибки или невозможности определить угол — возвращается оригинальное изображение.

    Параметры:
        image: входное изображение (массив NumPy)

    Возвращает:
        Выровненное изображение или исходное, если обработка не удалась
    """
    try:
        # Преобразование изображения в оттенки серого при необходимости
        if len(image.shape) == 3:
            grayscale = rgb2gray(image)
        else:
            grayscale = image.copy()

        # Определение угла наклона
        angle = determine_skew(grayscale)

        if angle is None:
            return image

        # Поворот изображения на найденный угол
        rotated = rotate(image, angle, resize=True) * 255

        # Приведение к формату uint8 для дальнейшей обработки
        rotated = rotated.astype(np.uint8)

        return rotated
    except Exception as e:
        print(f"Ошибка при выравнивании: {str(e)}")
        return image


def check_rotation(image):
    """
    Функция проверяет ориентацию изображения и при необходимости его поворачивает.

    Анализ проводится через обнаружение контуров текста и вычисление среднего угла их наклона.
    Также учитывается соотношение сторон изображения как дополнительный фактор.

    Параметры:
        image: входное изображение (массив NumPy)

    Возвращает:
        Повернутое изображение или оригинал, если изменений не требуется
    """
    try:
        # Преобразование в оттенки серого при необходимости
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # Адаптивная пороговая обработка для выделения текста
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
        )

        # Поиск контуров
        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # Фильтрация контуров по площади
        valid_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 100:  # Игнорируем слишком маленькие объекты
                valid_contours.append(contour)

        if valid_contours:
            # Получаем минимальные прямоугольники для контуров
            rects = [cv2.minAreaRect(contour) for contour in valid_contours]

            # Собираем углы наклона этих прямоугольников
            angles = []
            for rect in rects:
                angle = rect[2]
                # Нормализуем угол в диапазон [-45, 45] градусов
                if angle < -45:
                    angle += 90
                elif angle > 45:
                    angle -= 90
                angles.append(angle)

            # Вычисляем средний угол
            if angles:
                avg_angle = np.mean(angles)
                print(f"Средний угол наклона: {avg_angle}")

                # Если наклон значителен — поворачиваем изображение
                if abs(avg_angle) > 2:
                    print(f"Поворачиваю изображение на {avg_angle} градусов")
                    height, width = image.shape[:2]
                    center = (width // 2, height // 2)
                    rotation_matrix = cv2.getRotationMatrix2D(center, avg_angle, 1.0)
                    rotated = cv2.warpAffine(
                        image,
                        rotation_matrix,
                        (width, height),
                        flags=cv2.INTER_CUBIC,
                        borderMode=cv2.BORDER_REPLICATE,
                    )
                    return rotated

        # Дополнительная проверка: если ширина меньше высоты — возможно изображение повернуто
        height, width = image.shape[:2]
        if width < height:
            print("Поворачиваю изображение по соотношению сторон")
            return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)

        print("Поворот не требуется")
        return image

    except Exception as e:
        print(f"Ошибка при определении ориентации: {str(e)}")
        return image


def save_image(_img, _path):
    """
    Сохраняет изображение по указанному пути.

    Параметры:
        _img: изображение (массив NumPy)
        _path: путь для сохранения файла
    """
    try:
        cv2.imwrite(_path, _img)
    except Exception as e:
        print(f"Ошибка при сохранении изображения: {str(e)}")


if __name__ == "__main__":
    """
    Точка входа при запуске файла напрямую.

    Загружает пример изображения, применяет коррекцию наклона и сохраняет результат.
    """
    try:
        original_image_path = "../examples/sxJzw.jpg"
        image = io.imread(original_image_path)
        deskewed_image = deskew(image)
        save_image(deskewed_image, "../examples/deskewed_image.png")
    except Exception as e:
        print(f"Ошибка в главной программе: {str(e)}")
