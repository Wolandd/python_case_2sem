"""
Файл: imgFilter.py
Назначение: фильтрация и предварительная обработка изображений для улучшения распознавания текста.
"""

import cv2
import numpy as np
from PIL import Image
from .imgRotat import check_rotation, deskew


def apply_binarization(image, threshold_value=128):
    """
    Применяет бинаризацию изображения — преобразует в чёрно-белое по заданному порогу.

    Если входное изображение цветное, оно сначала конвертируется в оттенки серого.

    Параметры:
        image: входное изображение (массив NumPy)
        threshold_value: пороговое значение для бинаризации

    Возвращает:
        бинаризованное изображение
    """
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    _, binary_image = cv2.threshold(image, threshold_value, 255, cv2.THRESH_BINARY)
    return binary_image


def apply_canny_edge(image, low_threshold=50, high_threshold=150):
    """
    Применяет алгоритм Canny для выделения границ на изображении.

    Полезно для анализа структуры текста или таблиц.

    Параметры:
        image: входное изображение (массив NumPy)
        low_threshold: нижний порог чувствительности детектора
        high_threshold: верхний порог чувствительности детектора

    Возвращает:
        изображение с выделенными контурами
    """
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    edges = cv2.Canny(image, low_threshold, high_threshold)
    return edges


def apply_inverse(image):
    """
    Инвертирует цвета изображения — белое становится чёрным и наоборот.

    Может помочь при работе с тёмным текстом на светлом фоне.

    Параметры:
        image: входное изображение (массив NumPy)

    Возвращает:
        инвертированное изображение
    """
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    inverted_image = cv2.bitwise_not(image)
    return inverted_image


def apply_sharpen(image):
    """
    Увеличивает резкость изображения с помощью свёрточного ядра.

    Улучшает чёткость линий и символов, что помогает OCR-системам.

    Параметры:
        image: входное изображение (массив NumPy)

    Возвращает:
        изображение с повышенной резкостью
    """
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    kernel = np.array([[-1, -1, -1], [-1, 10, -1], [-1, -1, -1]])
    sharpened_image = cv2.filter2D(image, -1, kernel)
    return sharpened_image


def apply_morphological(image):
    """
    Применяет морфологические операции для улучшения формы текста.

    Убирает шумы и соединяет разорванные части букв.

    Параметры:
        image: входное изображение (массив NumPy)

    Возвращает:
        обработанное изображение
    """
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    kernel = np.ones((1, 1), np.uint8)
    morphological_image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    return morphological_image


def preprocess_image(image):
    """
    Объединяет несколько методов обработки изображения для улучшения качества текста.

    Последовательность шагов:
        1. Усиление резкости
        2. Морфологические преобразования
        3. Бинаризация
        4. Инверсия

    Подходит для подготовки изображений перед OCR-распознаванием.

    Параметры:
        image: входное изображение (PIL Image, массив NumPy или путь к файлу)

    Возвращает:
        полностью обработанное изображение (массив NumPy)
    """
    # Конвертация изображения в массив
    if isinstance(image, str):
        image = cv2.imread(image)
    elif isinstance(image, Image.Image):  # PIL Image
        image = np.array(image)
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    image = apply_sharpen(image)
    image = apply_morphological(image)
    image = apply_binarization(image)
    image = apply_inverse(image)

    return image


class ImageProcessor:
    def __init__(self):
        """
        Класс для комплексной обработки изображений.

        Хранит оригинальное, обработанное и отображаемое изображение.
        Предоставляет удобные методы для последовательной обработки.
        """
        self.original_image = None
        self.processed_image = None
        self.display_image = None
        self.rotation_angle = 0
        self.deskew_angle = 0

    def set_image(self, image):
        """
        Загружает изображение из разных источников: пути, PIL.Image или массива NumPy.

        Конвертирует его в формат OpenCV (BGR) и сохраняет как оригинал.
        """
        if isinstance(image, str):
            self.original_image = cv2.imread(image)
        elif isinstance(image, Image.Image):  # PIL Image
            self.original_image = np.array(image)
            if (
                len(self.original_image.shape) == 3
                and self.original_image.shape[2] == 3
            ):
                self.original_image = cv2.cvtColor(
                    self.original_image, cv2.COLOR_RGB2BGR
                )
        else:
            self.original_image = image.copy()

        self.display_image = self.original_image.copy()
        return self

    def preprocess(self):
        """
        Выполняет полную предобработку изображения:
            - Усиление резкости
            - Морфологические операции
            - Бинаризация
            - Инверсия

        Результат сохраняется как processed_image.
        """
        if self.original_image is None:
            raise ValueError("Изображение не загружено. Сначала вызовите set_image().")

        if len(self.original_image.shape) == 3:
            gray_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
        else:
            gray_image = self.original_image.copy()

        processed = apply_sharpen(gray_image)
        processed = apply_morphological(processed)
        processed = apply_binarization(processed)
        processed = apply_inverse(processed)

        self.processed_image = processed
        return self

    def rotate_for_display(self):
        """
        Поворачивает изображение для корректного отображения.

        Использует функции из imgRotat.py: проверка ориентации и выравнивание наклона.
        """
        if self.display_image is None:
            raise ValueError("Изображение не загружено. Сначала вызовите set_image().")

        try:
            rotated = check_rotation(self.display_image)
            if rotated is not None:
                self.display_image = rotated

            deskewed = deskew(self.display_image)
            if deskewed is not None:
                self.display_image = deskewed

            return self
        except Exception as e:
            print(f"Ошибка при повороте/выравнивании: {str(e)}")
            return self

    def get_processed_image(self):
        """
        Возвращает обработанное изображение для дальнейшей работы (например, OCR).
        """
        if self.processed_image is None:
            raise ValueError(
                "Обработанное изображение недоступно. Вызовите preprocess()."
            )
        return self.processed_image

    def get_display_image(self):
        """
        Возвращает изображение в формате RGB для отображения пользователю.
        """
        if self.display_image is None:
            raise ValueError("Изображение не загружено. Сначала вызовите set_image().")
        if len(self.display_image.shape) == 3 and self.display_image.shape[2] == 3:
            return cv2.cvtColor(self.display_image, cv2.COLOR_BGR2RGB)
        return self.display_image

    def get_original_image(self):
        """
        Возвращает исходное изображение.
        """
        if self.original_image is None:
            raise ValueError("Изображение не загружено. Сначала вызовите set_image().")
        return self.original_image


if __name__ == "__main__":
    """
    Точка входа при запуске файла напрямую.

    Загружает изображение и применяет различные методы обработки.
    Сохраняет результаты в папку ./examples/
    """
    image = cv2.imread("./examples/example.png", cv2.IMREAD_GRAYSCALE)
    binary_image = apply_binarization(image)
    canny_edge_image = apply_canny_edge(image)
    inversed_image = apply_inverse(image)
    sharp_image = apply_sharpen(image)
    morphological_image = apply_morphological(image)

    # Комбинированная обработка
    combined_image = apply_inverse(image)
    combined_image = apply_sharpen(combined_image)
    combined_image = apply_morphological(combined_image)
    combined_image = apply_binarization(combined_image)
    cv2.imwrite("./examples/combined_image.png", combined_image)

    # Сохранение результатов
    cv2.imwrite("./examples/binary_image.png", binary_image)
    cv2.imwrite("./examples/canny_edge_image.png", canny_edge_image)
    cv2.imwrite("./examples/inversed_image.png", inversed_image)
    cv2.imwrite("./examples/sharp_image.png", sharp_image)
    cv2.imwrite("./examples/morphological_image.png", morphological_image)
