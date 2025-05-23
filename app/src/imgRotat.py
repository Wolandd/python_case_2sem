import numpy as np
import cv2
from skimage import io
from skimage.transform import rotate
from skimage.color import rgb2gray
from deskew import determine_skew

def deskew(image):
    """
    Выравнивает изображение, определяя и исправляя угол наклона.
    
    Parameters:
    - image: Input image (numpy array)
    
    Returns:
    - Выровненное изображение или исходное в случае ошибки
    """
    try:
        # Конвертируем в оттенки серого для определения угла
        if len(image.shape) == 3:
            grayscale = rgb2gray(image)
        else:
            grayscale = image.copy()
        
        # Определяем угол наклона
        angle = determine_skew(grayscale)
        
        if angle is None:
            return image
            
        # Поворачиваем изображение
        rotated = rotate(image, angle, resize=True) * 255
        
        # Конвертируем обратно в uint8
        rotated = rotated.astype(np.uint8)
        
        return rotated
    except Exception as e:
        print(f"Error in deskew: {str(e)}")
        return image

def check_rotation(image):
    """
    Проверяет и корректирует ориентацию изображения.
    Использует анализ текста и соотношения сторон для определения ориентации.
    
    Parameters:
    - image: Input image (numpy array)
    
    Returns:
    - Повернутое или исходное изображение
    """
    try:
        # Конвертируем в оттенки серого
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Применяем адаптивную пороговую обработку
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                     cv2.THRESH_BINARY_INV, 11, 2)
        
        # Находим контуры
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Фильтруем контуры по размеру
        valid_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 100:  # Игнорируем слишком маленькие контуры
                valid_contours.append(contour)
        
        if valid_contours:
            # Находим минимальный ограничивающий прямоугольник для каждого контура
            rects = [cv2.minAreaRect(contour) for contour in valid_contours]
            
            # Анализируем ориентацию прямоугольников
            angles = []
            for rect in rects:
                angle = rect[2]
                # Нормализуем угол в диапазон [-45, 45]
                if angle < -45:
                    angle += 90
                elif angle > 45:
                    angle -= 90
                angles.append(angle)
            
            # Вычисляем средний угол наклона
            if angles:
                avg_angle = np.mean(angles)
                print(f"Average angle: {avg_angle}")
                
                # Если угол значительный, поворачиваем изображение
                if abs(avg_angle) > 2:  # Уменьшил порог для более точного выравнивания
                    print(f"Rotating image by {avg_angle} degrees")
                    height, width = image.shape[:2]
                    center = (width // 2, height // 2)
                    rotation_matrix = cv2.getRotationMatrix2D(center, avg_angle, 1.0)
                    rotated = cv2.warpAffine(image, rotation_matrix, (width, height),
                                           flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
                    return rotated
        
        # Проверяем соотношение сторон как дополнительный критерий
        height, width = image.shape[:2]
        if width < height:
            print("Rotating image based on aspect ratio")
            return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        
        print("No rotation needed")
        return image
        
    except Exception as e:
        print(f"Error in check_rotation: {str(e)}")
        return image

def save_image(_img, _path):
    """Сохраняет изображение по указанному пути"""
    try:
        cv2.imwrite(_path, _img)
    except Exception as e:
        print(f"Error saving image: {str(e)}")

if __name__ == "__main__":
    try:
        original_image_path = '../examples/sxJzw.jpg'
        image = io.imread(original_image_path)
        deskewed_image = deskew(image)
        save_image(deskewed_image, '../examples/deskewed_image.png')
    except Exception as e:
        print(f"Error in main: {str(e)}")