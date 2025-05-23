import os
import torch
from PIL import Image, ImageDraw, ImageFont
import requests
from io import BytesIO
from pathlib import Path
import re
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from src.textDetectiom import detect_text, get_bbox_coordinates, get_axis_aligned_bbox
from src.imgFilter import ImageProcessor
#from imgRotat import deskew

import json
from huggingface_hub import login
import cv2

model_path = "Qwen/Qwen2.5-VL-3B-Instruct"
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    device_map="cpu"
)
processor = AutoProcessor.from_pretrained(model_path)
processor.image_processor.do_resize = False
processor.image_processor.do_center_crop = False

# System Prompt 1: Document Type Classification
SYSTEM_PROMPT_1 = (
    "You are a document parser. Analyze the input image and identify the document type. "
    "Consider the following document types: passport, id_card, driver_license, certificate, contract, invoice, receipt, form, other. "
    "Look for specific visual elements, text patterns, and layout characteristics that are unique to each document type. "
    "Respond with ONLY the document type in JSON format: {'document_type': 'type'}"
)

# System Prompt 2: Form/Table Structure Detection
SYSTEM_PROMPT_2 = (
    "You are a document parser. Given the document type \"{document_type}\", analyze the input image to detect structured elements like forms or tables. "
    "Focus only on the most relevant fields for this document type. "
    "For example:"
    "- For passport: series, number, name, date_of_birth, place_of_birth, issue_date, authority"
    "- For id_card: id_number, name, date_of_birth, address, issue_date"
    "- For driver_license: license_number, name, date_of_birth, issue_date, expiry_date, categories"
    "- For certificate: certificate_number, recipient_name, issue_date, issuing_authority"
    "- For contract: contract_number, parties, date, subject, terms"
    "- For invoice: invoice_number, date, items, total_amount, tax"
    "- For receipt: date, items, total_amount, payment_method"
    "- For form: only the most relevant fields based on the form's purpose"
    "Respond ONLY with json of field type and bounding box of the segment."
)

SYSTEM_PROMPT_3 = (
    "You are a document parser. Analyze the input image and detect **all text**. "
    "1. Extract the text inside. \n"
    "2. Classify the text as \"handwrite\" or \"regular\". \n"
    "3. If applicable, link the text to a field from \"{fields}\". \n"
    "4. Use the document type \"{document_type}\" to contextualize the text (e.g., identify field names, formatting rules, or layout patterns specific to {document_type}). \n"
    "5. Preserve the original text's integrity and avoid assumptions not supported by the image. \n"
    "6. Remember that the text is in forms, you can find text around black lines of forms \n"
    "7. You shoul return all handwrited text on the form\n"
    "Return a JSON object with a list of handwritten text entries. When saving the result of using the bbox boundaries are the same as they were at the input of the request. Which I could then compare and understand in which bbox the handwritten text was. \n"
    "Use Format: {{\"handwritten_text\": [{{\"text\": '...', \"bbox\": {{\"x_min\":x_min, \"x_max\":x_max, \"y_min\":y_min, \"y_max\":y_max}}, \"classification\": \"type of text\", \"linked_field\": \"...\" (if applicable)}}]}}. "
    "EXAMPLE: "
    "Input: [Image of a {document_type} form with handwritten name 'John Doe' and date '01/01/2023'] "
    "Output: {{'handwritten_text': [{{'text': 'John Doe', \"bbox\": {{\"x_min\":x_min, \"x_max\":x_max, \"y_min\":y_min, \"y_max\":y_max}}, \"classification\": \"regular\", \"linked_field\": 'Name'}}, {{'text': '01/01/2023', \"bbox\": {{\"x_min\":x_min, \"x_max\":x_max, \"y_min\":y_min, \"y_max\":y_max}}, \"classification\": \"handwrite\", \"linked_field\": 'date'}}]}}"
    )

SYSTEM_PROMPT_4 = (
    "You are a document parser. Analyze the input image to detect **signatures or seals**. "
    "Return a JSON object with boolean flags and their locations (if found). "
    "Format: {'signature_detected': true/false, 'seal_detected': true/false, 'details': [...]}. "
    
    "EXAMPLE 1: "
    "Input: [Image of a contract with a signature]"
    "Output: {'signature_detected': true, 'seal_detected': false, 'details': [{'type': 'signature', \"bbox\": [x1, y1, x2, y2]}]} "
    "EXAMPLE 2: "
    "Input: [Image of a stamped certificate]"
    "Output: {'signature_detected': false, 'seal_detected': true, 'details': [{'type': 'seal', \"bbox\": [x1, y1, x2, y2]}]}"
)

SYSTEM_PROMPT_5 = (
    "You are a text correction expert. Your task is to correct any spelling, grammar, or syntax errors in the given text. "
    "Return ONLY the corrected text without any explanations or additional formatting. "
    "Preserve the original meaning and context of the text. "
    "If the text is already correct, return it as is. "
    "Example: "
    "Input: 'вносящчося каноudаmoм' "
    "Output: 'вносящиеся кандидатом'"
)

def inference(image, system_prompt, user_prompt="Analyze this document", max_new_tokens=16000):
    messages = [
        {
            "role": "system",
            "content": [
                {"type": "text", "text": system_prompt},
                {"type": "image_url", "image_url": {"url": image}}
            ]
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": user_prompt}
            ]
        }
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text],padding=True, return_tensors="pt")

    output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
    output_text = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]
    return output_text

def correct_text(text):
    """
    Corrects text using Qwen model
    """
    try:
        if not text or len(text.strip()) == 0:
            return text

        messages = [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": SYSTEM_PROMPT_5}
                ]
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"Correct this text: {text}"}
                ]
            }
        ]

        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], padding=True, return_tensors="pt")

        output_ids = model.generate(**inputs, max_new_tokens=100)
        generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
        corrected_text = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]
        
        # Очистка результата от возможных артефактов
        corrected_text = corrected_text.strip()
        if corrected_text.startswith('"') and corrected_text.endswith('"'):
            corrected_text = corrected_text[1:-1]
        
        return corrected_text
    except Exception as e:
        print(f"Error in text correction: {str(e)}")
        return text

def merge_model_outputs(document_type, fields_output, handwritten_output, signature_output):
    def clean_json_string(text):
        # Убираем лишние маркеры кода и парсим JSON
        text = text.replace("```json", "").replace("```", "").strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            print(f"⚠️ Ошибка при парсинге JSON: {e}")
            return {}

    merged = {
        "document_type": document_type.strip() if isinstance(document_type, str) else document_type,
        "structured_fields": clean_json_string(fields_output),
        "handwritten_texts": handwritten_output.get("handwritten_text", []),
        "signature_seal_info": signature_output
    }

    return merged

def process_document(img_input):
    """
    Основная функция для обработки документа:
    - Определение типа документа
    - Извлечение структуры (полей)
    - Распознавание текста
    - Обнаружение подписи/печати
    
    Parameters:
    - img_input: путь к файлу или numpy array с изображением
    
    Возвращает словарь с результатами всех этапов
    """
    try:
        # Обработка входных данных
        if isinstance(img_input, str):
            # Если передан путь к файлу
            with open(img_input, "rb") as image_file:
                import base64
                image_data = base64.b64encode(image_file.read()).decode('utf-8')
                image_url = f"data:image/png;base64,{image_data}"
            # Для OCR используем путь к файлу
            preds_easyOcr = detect_text(img_input)
        else:
            # Если передан numpy array
            # Сохраняем временно для base64
            temp_path = "temp_for_base64.png"
            cv2.imwrite(temp_path, img_input)
            with open(temp_path, "rb") as image_file:
                import base64
                image_data = base64.b64encode(image_file.read()).decode('utf-8')
                image_url = f"data:image/png;base64,{image_data}"
            
            # Создаем экземпляр ImageProcessor для предобработки
            processor = ImageProcessor()
            processor.set_image(img_input)
            processor.preprocess()
            processed_image = processor.get_processed_image()
            
            # Для OCR используем предобработанное изображение
            preds_easyOcr = detect_text(processed_image)
            
            # Масштабируем координаты bbox с предобработанного изображения на оригинальное
            original_height, original_width = img_input.shape[:2]
            processed_height, processed_width = processed_image.shape[:2]
            
            # Вычисляем коэффициенты масштабирования
            scale_x = original_width / processed_width
            scale_y = original_height / processed_height
            
            # Масштабируем координаты bbox
            for result in preds_easyOcr:
                bbox = result.get('bbox', {})
                if bbox:
                    bbox['x_min'] = int(bbox['x_min'] * scale_x)
                    bbox['y_min'] = int(bbox['y_min'] * scale_y)
                    bbox['x_max'] = int(bbox['x_max'] * scale_x)
                    bbox['y_max'] = int(bbox['y_max'] * scale_y)
            
            # Удаляем временный файл
            os.remove(temp_path)

        print("OCR Results:", preds_easyOcr)  # Debug print

        # Step 1: Document Type Classification
        doc_type = inference(image_url, SYSTEM_PROMPT_1).strip()

        # Step 2: Form/Table Structure Detection
        system_prompt_2 = SYSTEM_PROMPT_2.format(document_type=doc_type)
        fields_output = inference(image_url, system_prompt_2).strip()
        
        # Step 3: Text Recognition
        structured_output = inference(image_url, SYSTEM_PROMPT_3.format(document_type=doc_type, fields=fields_output))
        
        # Step 4: Signature/Seal Detection
        seal_output = inference(image_url, SYSTEM_PROMPT_4)

        # Преобразуем OCR результаты в нужный формат и корректируем текст
        ocr_results = []
        for result in preds_easyOcr:
            original_text = result.get('text', '')
            if original_text and len(original_text.strip()) > 0:
                corrected_text = correct_text(original_text)
                print(f"Original text: {original_text}")
                print(f"Corrected text: {corrected_text}")
            else:
                corrected_text = original_text

            ocr_results.append({
                'text': corrected_text,
                'bbox': result.get('bbox', {}),
                'classification': 'handwrite'
            })

        print("Processed OCR Results:", ocr_results)  # Debug print

        # Формируем итоговый результат
        result = {
            'document_type': doc_type,
            'structured_fields': fields_output,
            'handwritten_text': {
                'handwritten_text': ocr_results
            },
            'signature_seal': seal_output
        }

        print("Final Result:", result)  # Debug print
        return result

    except Exception as e:
        print(f"Error in process_document: {str(e)}")
        return {
            'document_type': 'unknown',
            'structured_fields': '{}',
            'handwritten_text': {'handwritten_text': []},
            'signature_seal': {'signature_detected': False, 'seal_detected': False, 'details': []}
        }

def clean_response(response):
    """Убирает Markdown-форматирование, лишние пробелы и обратные кавычки"""
    if not isinstance(response, str):
        return response
    
    # Удаление маркеров кода (например, ```json или ```)
    cleaned = re.sub(r'```(?:json|JSON)?', '', response).strip()
    
    # Удаление лишних escape-последовательностей (если они есть)
    cleaned = cleaned.replace('\\"', '"').replace("\\'", "'")
    
    return cleaned

def parse_json_safe(text):
    """Безопасно парсит строку в JSON, если возможно"""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text  # Если не JSON — возвращаем как есть

def merge_all_results(raw_data):
    """
    Корректно объединяет все части результата в один структурированный JSON.
    """
    try:
        # Очистка и парсинг каждой секции
        document_type = raw_data.get("document_type", "")
        if isinstance(document_type, str):
            document_type = parse_json_safe(clean_response(document_type))
        
        structured_fields = raw_data.get("structured_fields", "")
        if isinstance(structured_fields, str):
            structured_fields = parse_json_safe(clean_response(structured_fields))
        
        handwritten_text = raw_data.get("handwritten_text", "")
        if isinstance(handwritten_text, str):
            handwritten_text = parse_json_safe(clean_response(handwritten_text))
        
        signature_seal = raw_data.get("signature_seal", "")
        if isinstance(signature_seal, str):
            signature_seal = parse_json_safe(clean_response(signature_seal))

        # handwritten_text должен быть словарём с ключом handwritten_text
        if isinstance(handwritten_text, str):
            try:
                handwritten_text = json.loads(handwritten_text)
            except Exception:
                handwritten_text = {"handwritten_text": []}
        if not isinstance(handwritten_text, dict) or "handwritten_text" not in handwritten_text:
            handwritten_text = {"handwritten_text": []}

        merged_result = {
            "document_type": document_type,
            "structured_fields": structured_fields,
            "handwritten_text": handwritten_text,
            "signature_seal": signature_seal
        }

        print('MERGED JSON:', merged_result)  # Для отладки

        return merged_result
    except Exception as e:
        print(f"Error in merge_all_results: {str(e)}")
        return {
            "document_type": {"document_type": "unknown"},
            "structured_fields": {},
            "handwritten_text": {"handwritten_text": []},
            "signature_seal": {"signature_detected": False, "seal_detected": False, "details": []}
        }

def save_to_json(data, filename="output.json"):
    """Сохраняет данные в формате JSON в указанный файл"""
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    print(f"\n✅ Результат сохранен в {filename}")

if __name__ == "__main__":
    img_path = "../examples/binary_image.png"
    img_path = Image.open(img_path)
    
    result = process_document(img_path)

    with open('handwritten_text.txt','w') as f:
        f.write(result['handwritten_text'])
    # Сохранение результата в JSON-файл
    merged_result = merge_all_results(result)
    # merged_result['boxes'] = result['boxes']
    save_to_json(merged_result, "output.json")

    # Печать результата в консоль
    for key, value in result.items():
        print(f"\n{key.upper()}:")
        print(json.dumps(value, ensure_ascii=False, indent=2) if isinstance(value, dict) else value)
