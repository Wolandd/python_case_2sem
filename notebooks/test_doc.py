import os
import numpy as np
import pandas as pd
from tqdm.notebook import tqdm
from PIL import Image
import easyocr
from jiwer import cer, wer
from textblob import TextBlob
import string
from rapidfuzz import fuzz
import re

# Список путей к изображениям
image_paths = [
    "./examples/test_docs/1.png",
    "./examples/test_docs/2.jpg",
    "./examples/test_docs/3.png",
    "./examples/test_docs/4.png",
    "./examples/test_docs/5.png",
    "./examples/test_docs/6.jpg",
    "./examples/test_docs/7.jpg",
    "./examples/test_docs/8.jpg",
    "./examples/test_docs/9.jpeg",
    "./examples/test_docs/10.png"
]

# Соответствующие эталонные тексты
ground_truths = [
    "Реквизит Пояснение Основание Государственный герб РФ герб субъекта РФ герб геральдический знак муниципального образования Изображают на приказах органов власти и управления а также организаций в их подчинении Статья 9 Закона от 06.12.2011 № 402-ФЗ пункт 5.1 ГОСТ Р 70.97-2016 утвержденного приказом Росстандарта от 08.12.2016 № 2004-ст далее – ГОСТ Р 70.97-2016 Эмблема Отражают на бланках если это закреплено уставом или положением об организации Статья 9 Закона от 06.12.2011 № 402-ФЗ пункт 5.2 ГОСТ Р 70.97-2016 Товарный знак или знак обслуживания Отражают на бланках если это закреплено уставом или положением об организации Статья 9 Закона от 06.12.2011 № 402-ФЗ пункт 5.3 ГОСТ Р 70.97-2016",
    'Общество с ограниченной ответственностью «Альфа» ООО «Альфа» Начальнику отдела продаж Кондратьеву А.С. Распоряжение о выполнении работы 06.04.2023 Москва Уважаемый Александр Сергеевич! Вам необходимо разработать мотивацию менеджерам по продажам. Суть задания: составить программу мотивации менеджеров по продажам, включающую следующие условия: процент выполнения плана, при котором начинает работать программа; индивидуальные и коллективные показатели эффективности отдела; алгоритм контроля за соблюдением программы мотивации. Срок: 20.04.2020 Директор А.В. Львов С распоряжением ознакомлен Кондратьев А.С. 06.04.2023',
    'ООО «Наша фирма» УТВЕРЖДАЮ Генеральный директор АКТ М.Н. Федоров 11.11.2011 № 7 11.11.2014 г. Йошкар-Ола О выделении документов к уничтожению На основании перечня документов № 12 «Документы по учету товарно-материальных ценностей со сроком хранения 3 года» отобраны к уничтожению как не имеющие практического значения и научно-исторической ценности документы: Заголовок дела Дата дела или край- ние даты дел Кол-во дел (томов) Срок хранения и №№ статей по перечню Примечание Итого дел за годы. Описи дел за годы утверждены генеральным директором. Председатель комиссии И. И. Иванов (подпись) Члены комиссии В. В. Петров (подпись) П. П. Потапов (подпись)',
    'ООО «Наша фирма» УТВЕРЖДАЮ Генеральный директор АКТ М.Н. Федоров 11.11.2011 № 7 11.11.2014 г. Йошкар-Ола О выделении документов к уничтожению На основании перечня документов № 12 «Документы по учету товарно-материальных ценностей со сроком хранения 3 года» отобраны к уничтожению как не имеющие практического значения и научно-исторической ценности документы: Заголовок дела Дата дела или край- ние даты дел Кол-во дел (томов) Срок хранения и №№ статей по перечню Примечание Итого дел за годы. Описи дел за годы утверждены генеральным директором. Председатель комиссии И. И. Иванов (подпись) Члены комиссии В. В. Петров (подпись) П. П. Потапов (подпись)',
    'ДОГОВОР ДАРЕНИЯ КВАРТИРЫ Город (число, месяц, год) Я, гр. _____, __ года рождения, место рождения: _____, гражданство: _____, пол: _____, паспорт серия № _____, выданный года, код подразделения _____, зарегистрированный(ая) по адресу: _____, с одной стороны, и гр. _____, __ года рождения, место рождения: _____, гражданство: _____, пол: _____, паспорт серия № _____, выданный года, код подразделения _____, зарегистрированный(ая) по адресу: _____, с другой стороны, находясь в здравом уме и твердой памяти, действуя добровольно, заключили настоящий договор о нижеследующем: 1. Я, _____, подарил своему сыну/дочери/брату/сестре/внуку/внучке (заполняется для близких родственников) (ФИО) принадлежащую мне по праву собственности КВАРТИРУ, находящуюся по адресу: состоящую из ___ комнат общей площадью ___ (_____) кв.м. Кадастровый номер объекта - 2. Указанная квартира принадлежит на праве собственности на основании (заполняются основания возникновения права, указанные в выписке ЕГРН или свидетельстве о праве собственности): - Договора купли-продажи №1 от 01.06.2019 года - Акта приема-передачи №1 к договору от 01.06.2019 года Или - Свидетельства о праве собственности серия __ № выдано _____, кем: 3. Я, (ФИО Одаряемого) указанную квартиру в дар от (ФИО Дарителя) принимаю. 4. Даритель гарантирует, что он заключает настоящий договор не вследствие стечения тяжелых обстоятельств на крайне невыгодных для себя условиях и настоящий договор не является для них кабальной сделкой. 5. Право собственности на указанную квартиру возникает у Одаряемого с момента регистрации перехода права собственности в Управлении Федеральной службы государственной регистрации, кадастра и картографии. 6. Одаряемый в соответствии с законом несет бремя содержания указанной квартиры, а также бремя содержания общего имущества собственников помещений в многоквартирном доме. 7. Содержание статей 17, 30 Жилищного Кодекса Российской Федерации, 167, 209, 223, 288, 292 и 572 Гражданского кодекса Российской Федерации сторонам известно.',
    'ПРАВИТЕЛЬСТВО МОСКВЫ КОМИТЕТ ПО АРХИТЕКТУРЕ И ГРАДОСТРОИТЕЛЬСТВУ ГОРОДА МОСКВЫ (МОСКОМАРХИТЕКТУРА) 09 ОКТ 2013 13 / 006 / 7 номер ПАСПОРТ КОЛОРИСТИЧЕСКОГО РЕШЕНИЯ ФАСАДОВ ЗДАНИЙ, СТРОЕНИЙ, СООРУЖЕНИЙ В ГОРОДЕ МОСКВЕ АДРЕС ОБЪЕКТА: Кирпичная ул., д.35 РАЙОН ГОРОДА МОСКВЫ Соколиная гора АДМИНИСТРАТИВНЫЙ ОКРУГ ГОРОДА МОСКВЫ ВАО Заместитель председателя МОСКОМАРХИТЕКТУРЫ Т.Н.Гук',
    'ООО «Строительно-Экспертная Компания» Свидетельство об аттестации лаборатории неразрушающего контроля № 41А180289 Свидетельство СРО НП «ЭнергоАудит 31» № СРО-Э-031/375А Заказчик: ООО ИСК «Русь» ЭНЕРГЕТИЧЕСКИЙ ПАСПОРТ ЗДАНИЯ «Комплекс многоквартирных многоэтажных домов с помещениями общественного назначения, автостоянками, контрольно-пропускными пунктами, трансформаторными подстанциями по ул. Дачная, 21 в Заельцовском районе г. Новосибирска. Многоэтажный многоквартирный дом №1 (по генплану) со встроенно-пристроенными помещениями общественного назначения, подземная автостоянка №2 (по генплану), трансформаторные подстанции №4, 4/1 (по генплану)». Секции А,Б, пристроенные помещения» Директор ООО «Строительно-Экспертная Компания» Н.А. Проталинский г. Новосибирск 2016',
    'РОССИЙСКАЯ ФЕДЕРАЦИЯ Настоящий диплом свидетельствует о том что диплом о среднем профессиональном образовании Квалификация 102208 0000205 ДОКУМЕНТ ОБ ОБРАЗОВАНИИ И О КВАЛИФИКАЦИИ Регистрационный номер Дата выдачи освоил(а) образовательную программу среднего профессионального образования и успешно прошел(шла) государственную итоговую аттестацию Решение Государственной экзаменационной комиссии Председатель Государственной экзаменационной комиссии Руководитель образовательной организации М.П.',
    'Ассоциация «Российские алмазы» Совет директоров РЕШЕНИЕ 03.03.2004 № 5 Санкт-Петербург О подготовке к аукциону На основании координационного плана от 15.01.2004 г. Совет директоров ассоциации «Российские алмазы» РЕШИЛ: 1. Назначить ответственного представителем ассоциации на аукционе начальника отдела маркетинга В. В. Петрова. 2. В. В. Петрову подготовить предложения по составу делегации и представить на рассмотрение Совету до 10.03.2004. 3. Начальнику отдела продаж А. А. Иванову подготовить уточненный ассортимент ювелирных изделий и представить на рассмотрение Совету до 10.03.2004. Председатель Совета подпись Н. Н. Сидоров Члены Совета подпись А. Н. Чернов подпись О. И. Краснов',
    'ООО «ПОЛЁТ» ИНН 7777777777 КПП 555555555 Почтовый адрес: г. Санкт-Петербург, 1-я линия ВО 2019 г. Главному врачу поликлиники № 22 Просим вас подтвердить факт выдачи больничного листа № 12345678, оформленного на имя Петрова Петра Васильевича с 23 февраля по 8 марта 2019 г. врачом Васильевым В.П. Начальник отдела кадров Симонова М.А.'
]
# Функция нормализации текста (без TextBlob)
def normalize_text(text):
    text = text.lower()
    # Сохраняем точки в сокращениях (например, "г.", "ул.")
    text = re.sub(r'(?<!\w)\.(?!\w)', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation.replace('.', '')))
    return ' '.join(text.split())

# Основная функция для обработки документов
def evaluate_documents(image_paths, ground_truths):
    results = []
    for img_path, gt_text in tqdm(zip(image_paths, ground_truths), total=len(image_paths), desc="Processing documents"):
        try:
            # Проверка существования файла
            if not os.path.exists(img_path):
                print(f"❌ Файл не найден: {img_path}")
                continue

            # Загрузка изображения
            img = Image.open(img_path)
            img_array = np.array(img)

            # OCR
            reader = easyocr.Reader(['ru'], gpu=False)
            ocr_results = reader.readtext(img_array, detail=0, paragraph=False)
            if not ocr_results:
                print(f"❌ OCR не распознал текст для: {img_path}")
                continue

            pred_text = " ".join(sorted(ocr_results, key=lambda x: x[0][0]))  # Сортировка по координатам

            # Нормализация
            gt_text_norm = normalize_text(gt_text)
            pred_text_norm = normalize_text(pred_text)

            # Вычисление метрик
            cer_score = cer(gt_text_norm, pred_text_norm)
            wer_score = wer(gt_text_norm, pred_text_norm)
            fuzzy_match = fuzz.token_sort_ratio(gt_text_norm, pred_text_norm) / 100

            results.append({
                'document': os.path.basename(img_path),
                'cer': cer_score,
                'wer': wer_score,
                'fuzzy_match': fuzzy_match
            })

        except Exception as e:
            print(f"❌ Ошибка при обработке {img_path}: {str(e)}")

    if not results:
        raise ValueError("Не удалось обработать ни одно изображение. Проверьте данные.")

    return pd.DataFrame(results)

# Вызов функции
metrics_df = evaluate_documents(image_paths, ground_truths)
print(f"CER: {metrics_df['cer'].mean():.4f}")
print(f"WER: {metrics_df['wer'].mean():.4f}")
print(metrics_df['cer'])

# ANSWER:

# CER: 0.7108
# WER: 0.9404
# 0    0.700893
# 1    0.620805
# 2    0.765922
# 3    0.760331
# 4    0.729111
# 5    0.725410
# 6    0.672897
# Name: cer, dtype: float64
# 0    0.893204
# 1    0.942857
# 2    0.934426
# 3    1.021739
# 4    0.926829
# 5    1.040000
# 6    0.823529
