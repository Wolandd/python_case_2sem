import streamlit as st
import json
import time
from PIL import Image, ImageDraw, ImageFont
import io
from datetime import datetime
import numpy as np
import cv2
from skimage import io as skio
from skimage.transform import rotate
from skimage.color import rgb2gray
from deskew import determine_skew
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))
from src.infer import process_document, merge_all_results, save_to_json
from src.imgRotat import check_rotation, deskew
from src.imgFilter import ImageProcessor

def parse_annotations(json_data):
    # Parse the document type
    document_type = json_data.get('document_type', {})
    if isinstance(document_type, str):
        try:
            document_type = json.loads(document_type)
        except json.JSONDecodeError:
            document_type = {"document_type": "unknown"}
    
    # Parse the structured fields
    structured_fields = json_data.get('structured_fields', {})
    if isinstance(structured_fields, str):
        try:
            structured_fields = json.loads(structured_fields)
        except json.JSONDecodeError:
            structured_fields = {}
    
    # Parse the handwritten text
    handwritten_text = json_data.get('handwritten_text', {})
    if isinstance(handwritten_text, str):
        try:
            handwritten_text = json.loads(handwritten_text)
        except json.JSONDecodeError:
            handwritten_text = {"handwritten_text": []}
    
    # Parse the signature and seal information
    signature_seal = json_data.get('signature_seal', {})
    if isinstance(signature_seal, str):
        try:
            signature_seal = json.loads(signature_seal)
        except json.JSONDecodeError:
            signature_seal = {"signature_detected": False, "seal_detected": False, "details": []}
    
    # Combine all the data
    result = {
        'document_type': document_type,
        'structured_fields': structured_fields,
        'handwritten_texts': handwritten_text.get('handwritten_text', []),
        'signature_seal': signature_seal
    }
    
    print("Parsed annotations:", result)  # Debug print
    return result

# Настройка конфигурации страницы
st.set_page_config(
    page_title="Обработка документов",
    page_icon="📄",
    layout="wide"
)

# Добавление пользовательского CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
    }
    .uploaded-image {
        max-width: 100%;
        height: auto;
    }
    .processing-time {
        background-color: #262730;
        color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

#rotator
# def rotate_photo(img):
#     img = Image.open(img)
#     img = np.array(img)
#     return rotator.rotate_image(img)

# Заголовок и шапка
st.title("Обработка документов")
st.markdown("---")

# Секция загрузки файла
st.header("Загрузка документа")
uploaded_file = st.file_uploader("Выберите файл изображения", type=['png', 'jpg', 'jpeg', 'bmp', 'tiff', 'webp'])
if uploaded_file is not None:
    # Инициализация процессора изображений
    processor = ImageProcessor()
    
    # Загрузка и установка изображения
    image = Image.open(uploaded_file)
    processor.set_image(image)
    
    # Применяем препроцессинг для модели
    processor.preprocess()
    
    # Получаем препроцессированное изображение для модели
    processed_image_for_model = processor.get_processed_image()
    
    # Поворачиваем изображение только для отображения
    processor.rotate_for_display()
    
    # Получаем изображение для отображения
    display_image = Image.fromarray(processor.get_display_image())
    
    # Отображение оригинального изображения в начале
    st.subheader("Оригинальное изображение (после коррекции поворота)")
    st.image(display_image, caption='Загруженное изображение', use_column_width=True)
    st.markdown("---")
    
    # Начало отсчета времени
    start_time = time.time()
    
    # Обработка изображения с помощью ML
    with st.spinner('Обработка изображения...'):
        try:
            # Обработка документа с препроцессированным изображением
            result = process_document(processed_image_for_model)
            print("Process Document Result:", result)  # Отладочная информация
            
            merged_result = merge_all_results(result)
            print("MERGED JSON:", merged_result)  # Отладочная информация
            
            # Расчет времени обработки
            total_processing_time = time.time() - start_time
            
            # Парсинг данных
            parsed_data = parse_annotations(merged_result)
            print("Parsed Data:", parsed_data)  # Отладочная информация
            
            # Создаем копии изображения для отрисовки
            processed_image = display_image.copy()
            boxes_image = display_image.copy()

            # Создаем объекты для рисования
            draw = ImageDraw.Draw(processed_image)
            boxes_draw = ImageDraw.Draw(boxes_image)
            
            # Получение размеров изображения
            img_width, img_height = display_image.size
            print(f"Image size: {img_width}x{img_height}")
            
            # Получаем размеры препроцессированного изображения для масштабирования координат
            proc_height, proc_width = processed_image_for_model.shape[:2]
            print(f"Processed image size: {proc_width}x{proc_height}")
            
            # Отрисовка боксов для рукописного текста
            if 'handwritten_texts' in parsed_data and parsed_data['handwritten_texts']:
                print(f"Found {len(parsed_data['handwritten_texts'])} handwritten texts")  # Debug print
                for text_item in parsed_data['handwritten_texts']:
                    try:
                        bbox = text_item.get('bbox', {})
                        print(f"Processing bbox: {bbox}")  # Отладочная информация
                        
                        if not bbox:
                            print("Skipping item without bbox")
                            continue
                        
                        # Масштабируем координаты из препроцессированного изображения в размеры отображаемого
                        x_min = int(bbox.get('x_min', 0) * img_width / proc_width)
                        y_min = int(bbox.get('y_min', 0) * img_height / proc_height)
                        x_max = int(bbox.get('x_max', 0) * img_width / proc_width)
                        y_max = int(bbox.get('y_max', 0) * img_height / proc_height)
                        
                        # Проверяем, что координаты не выходят за границы изображения
                        x_min = max(0, min(x_min, img_width))
                        y_min = max(0, min(y_min, img_height))
                        x_max = max(0, min(x_max, img_width))
                        y_max = max(0, min(y_max, img_height))
                        
                        print(f"Drawing box at: ({x_min}, {y_min}, {x_max}, {y_max})")  # Отладочная информация
                        
                        # Проверка валидности размеров
                        if x_max <= x_min or y_max <= y_min:
                            print(f"Skipping invalid bbox: {bbox}")
                            continue
                            
                        text = text_item.get('text', '')
                        if not text or len(text.strip()) == 0:
                            text = "Текст"  # Добавляем текст по умолчанию
                        
                        # Рисуем белый фон и синий бокс на processed_image
                        draw.rectangle([x_min, y_min, x_max, y_max], fill="white")
                        draw.rectangle([x_min, y_min, x_max, y_max], outline="blue", width=3)
                        
                        # Рисуем только синий бокс на boxes_image
                        boxes_draw.rectangle([x_min, y_min, x_max, y_max], outline="blue", width=3)
                        
                        # Добавление текста с автоматическим подбором размера
                        try:
                            font_size = 1
                            font = ImageFont.truetype("Arial", font_size)
                            while True:
                                # Получаем размеры текста с текущим размером шрифта
                                text_width = draw.textlength(text, font=font)
                                bbox_text = font.getbbox(text)
                                text_height = bbox_text[3] - bbox_text[1]
                                
                                # Проверяем, помещается ли текст в бокс
                                if text_width >= (x_max - x_min) * 0.9 or text_height >= (y_max - y_min) * 0.9:
                                    break
                                font_size += 1
                                font = ImageFont.truetype("Arial", font_size)
                            
                            # Уменьшаем размер шрифта на 10% для комфортного отображения
                            font_size = max(1, int((font_size - 1) * 0.9))
                            font = ImageFont.truetype("Arial", font_size)
                            
                            # Центрирование текста
                            bbox_text = font.getbbox(text)
                            text_width = draw.textlength(text, font=font)
                            text_height = bbox_text[3] - bbox_text[1]
                            
                            # Вычисляем координаты для центрирования
                            text_x = x_min + (x_max - x_min - text_width) / 2
                            text_y = y_min + (y_max - y_min - text_height) / 2
                            
                            # Отрисовка текста
                            draw.text((text_x, text_y), text, fill="black", font=font)
                        except Exception as e:
                            print(f"Error processing text: {str(e)}")
                            # Если не удалось загрузить Arial, используем дефолтный шрифт
                            font = ImageFont.load_default()
                            text_width = draw.textlength(text, font=font)
                            text_x = x_min + (x_max - x_min - text_width) / 2
                            text_y = y_min + (y_max - y_min - font.getsize(text)[1]) / 2
                            draw.text((text_x, text_y), text, fill="black", font=font)
                        
                    except Exception as e:
                        print(f"Error processing bbox: {str(e)}")
                        continue

            # Отрисовка подписи, если она обнаружена
            if parsed_data.get('signature_seal', {}).get('signature_detected', False):
                for detail in parsed_data['signature_seal'].get('details', []):
                    if 'bbox' in detail:
                        bbox = detail['bbox']
                        # Масштабируем координаты из препроцессированного изображения в размеры отображаемого
                        x_min = int(bbox['x_min'] * img_width / proc_width)
                        y_min = int(bbox['y_min'] * img_height / proc_height)
                        x_max = int(bbox['x_max'] * img_width / proc_width)
                        y_max = int(bbox['y_max'] * img_height / proc_height)
                        
                        # Проверяем, что координаты не выходят за границы изображения
                        x_min = max(0, min(x_min, img_width))
                        y_min = max(0, min(y_min, img_height))
                        x_max = max(0, min(x_max, img_width))
                        y_max = max(0, min(y_max, img_height))
                        
                        # Проверка валидности размеров
                        if x_max <= x_min or y_max <= y_min:
                            continue
                        
                        # Рисуем красный бокс для подписи
                        boxes_draw.rectangle([x_min, y_min, x_max, y_max], outline="red", width=3)
                        
                        # Добавляем текст "Подпись" над боксом
                        try:
                            font = ImageFont.truetype("Arial", 12)
                        except:
                            font = ImageFont.load_default()
                        
                        text = "Подпись"
                        text_width = boxes_draw.textlength(text, font=font)
                        text_height = 15
                        
                        # Проверяем, не выходит ли текст за границы изображения
                        if y_min - text_height - 2 >= 0 and x_min + text_width + 4 <= img_width:
                            # Рисуем белый фон для текста
                            boxes_draw.rectangle(
                                [x_min, y_min - text_height - 2, x_min + text_width + 4, y_min],
                                fill="white"
                            )
                            
                            # Рисуем текст
                            boxes_draw.text((x_min + 2, y_min - text_height), text, fill="red", font=font)
                        else:
                            # Если текст не помещается сверху, рисуем его снизу
                            if y_max + text_height + 2 <= img_height and x_min + text_width + 4 <= img_width:
                                boxes_draw.rectangle(
                                    [x_min, y_max, x_min + text_width + 4, y_max + text_height + 2],
                                    fill="white"
                                )
                                boxes_draw.text((x_min + 2, y_max + 2), text, fill="red", font=font)

            # Отображение результатов
            objects_count = len(parsed_data.get('handwritten_texts', []))
            if parsed_data.get('signature_seal', {}).get('signature_detected', False):
                objects_count += len(parsed_data['signature_seal'].get('details', []))
            st.success(f'Обработка завершена! Найдено объектов: {objects_count}')
            
            # Добавление кнопки переключения
            st.subheader("Распознанный рукописный текст")
            show_processed = st.checkbox("Показать обработанное изображение", value=False)
            
            # Отображение изображения в зависимости от состояния кнопки
            if show_processed:
                st.image(processed_image, caption='Печатный текст вместо рукописного', use_column_width=True)
            else:
                st.image(boxes_image, caption='Найденный текст', use_column_width=True)
            
            st.header("Результаты обработки")
            
            # Display JSON in a collapsible expander
            st.subheader("JSON ответ")
            with st.expander("Нажмите для просмотра JSON"):
                st.json(parsed_data)
                
            # Add download button for JSON
            json_str = json.dumps(parsed_data, indent=2)
            st.download_button(
                label="Скачать JSON",
                data=json_str,
                file_name="processing_results.json",
                mime="application/json"
            )
            
            # Move processing time to the bottom
            st.markdown("""
            <div class="processing-time">
                <h3>Время обработки</h3>
                <p>Общее время обработки: {:.2f} секунд</p>
            </div>
            """.format(total_processing_time), unsafe_allow_html=True)
            
        except Exception as e:
            st.error(f"Произошла ошибка при обработке изображения: {str(e)}")
            print(f"Error details: {str(e)}")

# Подвал
st.markdown("---")
st.markdown("Команда The Boys") 