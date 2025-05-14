import streamlit as st
import json
import time
from PIL import Image, ImageDraw, ImageFont
import io
from datetime import datetime
import numpy as np
#import src.imgRotator as rotator

def parse_annotations(json_data):
    # Parse the structured fields
    structured_fields = json_data['structured_fields']
    
    # Parse the handwritten text
    handwritten_text = json_data['handwritten_text']
    
    # Parse the signature and seal information
    signature_seal = json_data['signature_seal']
    
    # Combine all the data
    result = {
        'document_type': json.loads(json_data['document_type'].replace("'", '"')),
        'structured_fields': structured_fields,
        'handwritten_texts': handwritten_text['handwritten_texts'],
        'signature_seal': signature_seal
    }
    
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
    # Отображение загруженного изображения
    image = Image.open(uploaded_file)
    
    # Отображение оригинального изображения в начале
    st.subheader("Оригинальное изображение")
    st.image(image, caption='Загруженное изображение', use_column_width=True)
    st.markdown("---")
    
    # Начало отсчета времени
    start_time = time.time()
    
    # Имитация обработки ML
    with st.spinner('Обработка изображения...'):
        # Имитация времени обработки ML
        #time.sleep(13.65)  # Имитация обработки ML
        
        # Расчет времени обработки
        total_processing_time = time.time() - start_time
        
        # Загрузка и парсинг данных из JSON файла (мок ответ от ML)
        with open('output2.json', 'r', encoding='utf-8') as f:
            json_data = json.load(f)
        parsed_data = parse_annotations(json_data)
        
        # Создание обработанного изображения
        processed_image = image.copy()
        boxes_image = image.copy()

        draw = ImageDraw.Draw(processed_image)
        boxes_draw = ImageDraw.Draw(boxes_image)
        
        # Получение размеров изображения
        img_width, img_height = image.size
        
        # Отрисовка боксов для рукописного текста
        for text_item in parsed_data['handwritten_texts']:
            bbox = text_item['bbox']  # [left, top, right, bottom] в пикселях
            text = text_item['text']
            
            # Рисуем белый фон и текст
            draw.rectangle([bbox[0], bbox[1], bbox[2], bbox[3]], fill="white")
            draw.rectangle([bbox[0], bbox[1], bbox[2], bbox[3]], outline="blue", width=2)
            boxes_draw.rectangle([bbox[0], bbox[1], bbox[2], bbox[3]], outline="blue", width=2)
            
            # Добавление текста
            # Вычисление размера шрифта, который поместится в бокс
            font_size = 1
            font = ImageFont.truetype("Arial.ttf", font_size)
            while True:
                bbox_text = font.getbbox(text)
                text_width = bbox_text[2] - bbox_text[0]
                text_height = bbox_text[3] - bbox_text[1]
                if text_width >= (bbox[2] - bbox[0]) * 0.9 or text_height >= (bbox[3] - bbox[1]) * 0.9:
                    break
                font_size += 1
                font = ImageFont.truetype("Arial.ttf", font_size)
            font_size = max(1, int((font_size - 1) * 0.9))  # Уменьшаем на 1 и делаем на 10% меньше
            font = ImageFont.truetype("Arial.ttf", font_size)
            
            # Центрирование текста
            bbox_text = font.getbbox(text)
            text_width = bbox_text[2] - bbox_text[0]
            text_height = bbox_text[3] - bbox_text[1]
            
            # Точное центрирование с учетом границ текста
            text_x = bbox[0] + (bbox[2] - bbox[0] - text_width) / 2 - bbox_text[0]
            text_y = bbox[1] + (bbox[3] - bbox[1] - text_height) / 2 - bbox_text[1]
            
            draw.text((text_x, text_y), text, fill="black", font=font)
        
        # Отрисовка подписи, если она обнаружена
        if parsed_data['signature_seal']['signature_detected']:
            for detail in parsed_data['signature_seal']['details']:
                if 'bbox' in detail:
                    bbox = detail['bbox']
                    draw.rectangle([bbox[0], bbox[1], bbox[2], bbox[3]], outline="blue", width=2)
                    boxes_draw.rectangle([bbox[0], bbox[1], bbox[2], bbox[3]], outline="blue", width=2)
        
        # Отображение результатов
        objects_count = len(parsed_data['handwritten_texts'])
        if parsed_data['signature_seal']['signature_detected']:
            objects_count += len(parsed_data['signature_seal']['details'])
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

# Подвал
st.markdown("---")
st.markdown("T1 case by Meowching learning") 