# Research dock
## Basic detection
- Table transformer (2021) (MIT LICENSE)
    Links
    - github - https://github.com/microsoft/table-transformer
    - paper - https://arxiv.org/abs/2110.00061
    - huggingface - https://huggingface.co/datasets/bsmock/pubtables-1m (датасет)
    - papers_with_code - https://paperswithcode.com/dataset/pubtables-1m (датасет)
    
    моделька microsoft для детекции таблиц, обученная на их датасете PubTables-1M (вместе с ней и представленная)
    Есть две версии модели:
    - Detr r18 - для Table detection (https://huggingface.co/microsoft/table-transformer-detection)
    - TATR-v1 - для Table structure recognition (https://huggingface.co/microsoft/table-transformer-structure-recognition)
    
    Тестовый ноутбук - https://colab.research.google.com/github/NielsRogge/Transformers-Tutorials/blob/master/Table%20Transformer/Inference_with_Table_Transformer_(TATR)_for_parsing_tables.ipynb#scrollTo=CBUsZT0OD9-3
    + ноутбук с нашими тестами - /notebooks/TATRv1.ipynb
    По результатам тестов выводы следующие:
    - Детекция таблиц по горизонтали не точное (Можно тюнить отступ от краёв детекции, однако, это не сильно спасает). Вертикальная детекция хороша.
    - Детекция таблиц на основе исправленной картинки улучшает детекцию по горизонтале, но ухудшает вертикальную.
    - Easy ocr работает удовлетворительно при хорошей чёткости текста + туда уже встроен препроцессор для текста.
    
- PaddleOCR (2021) (Apache License 2.0)
    Links
    - github - https://github.com/PaddlePaddle/PaddleOCR
    - documentation - https://paddlepaddle.github.io/PaddleOCR/latest/en/index.html
    
    Фреймворк для всего что связано с текстом на изображениях.
    Поддерживает русский язык
    - Нормальная детекция текста [3]
    - Плохое распознавание русского текста OCR 
    - Слабая детекция таблиц
    
    Тестовый ноут для задачи детекции и распознавания текста - https://colab.research.google.com/drive/1TYhLsOYW4nVfX5NP8Fi-O1QU0_ndj_ik?usp=sharing#scrollTo=kXPFPQrmnrdA
    
- Easy OCR 
    Links
    - github - https://github.com/JaidedAI/EasyOCR
    - documentation - https://github.com/JaidedAI/EasyOCR/blob/master/README.md
    
    Поддержка русского языка, хорошая детекция текста, нормальное распознование текста.
    Хороший бейзлайн для детекции текста[3,4]
    Detector CRAFT, Recognition ResNet+LSTM+CTC - нормально (см документацию для схемы)

- Tesseract OCR
    Links
    - github - https://github.com/tesseract-ocr/tesseract
    - documentation - https://tesseract-ocr.github.io/tessdoc/Installation.html
    
    Поддержка русского языка, очень сильно зависит от качества изображения.
    То есть работает кое-как адекватно только с идеальным текстом.[3,4]
    
- MMOCR
    Links
    - github - https://github.com/open-mmlab/mmocr
    - documentation - https://mmocr.readthedocs.io/en/latest/
    
    Имплементированно много моделей, но все обученны на eng-only или ch-only датасетах, то есть нужно обучение
    Очень долгая сборка (выявлено по итогам теста колаба (так и не собралось за 30 минут на gpu t4 системе)
    
    Туториал ноут - https://colab.research.google.com/github/open-mmlab/mmocr/blob/dev-1.x/demo/tutorial.ipynb
    
### HTR 
Здесь собраны проекты, реализованные детекторы конкретно для HTR
- https://github.com/NastyBoget/hrtr - уже выполненый проект, нет весов
- https://www.kaggle.com/code/constantinwerner/simple-and-efficient-transformer-htr#Test - выполненный проект, нужно даунгрейдить пайтон
- https://www.kaggle.com/code/coolaccount227/trocr-cyrillic/notebook - более актуальный детектор, весов нет; используется trocr 
- https://huggingface.co/kazars24/trocr-base-handwritten-ru - та же модель что и в пункте выше
Последние два по перфомансу в теории лучшие. Протестить особо не вышло, тоже из за проблем с зависимостями, скорее всего легко фиксится даунгрейдом пайтона
    
## Foundational Models
- Qwen 2.5 vl 2024
    Links
    - github - https://github.com/QwenLM/Qwen2.5-VL
    - paper - https://arxiv.org/abs/2502.13923
    - huggingface - https://huggingface.co/collections/Qwen/qwen25-vl-6795ffac22b334a837c0f9a5
    
    3b модель потребляет около 8 гб памяти в fp16 весах. Квантованная 7b тоже в теории влезет
    ноутбук с doc parse - https://github.com/QwenLM/Qwen2.5-VL/blob/main/cookbooks/document_parsing.ipynb (парсит на английском, нужно по работать над промптами)
    (см notebooks/Qwen2.5-vl-3b-dock-parce-test.ipynb)
    ноутбук с ocr - https://github.com/QwenLM/Qwen2.5-VL/blob/main/cookbooks/ocr.ipynb (завёлся, но механизм flash-attn2 не работает - старая архитектура gpu)
    
- Florence 2 (2023)
    Links
    - example - https://huggingface.co/microsoft/Florence-2-large/blob/main/sample_inference.ipynb
    - paper - https://arxiv.org/abs/2311.06242
    - huggingface - https://huggingface.co/microsoft/Florence-2-large
    
    VLM, а по факту первая по настоящему Foundational model в CV. В качестве vision encoder юзают DaViT. 
    По метрикам проигрывает классическим детекторам, но может немного хендлить контекст и выполнять сразу несколько задач (распознать по отдельности рукописный и печатный текст, сделать layout анализ) Из за слабых детект возможностей, для нашей задачи плохо подходит (см examples/misc/Florence2-test.png)
    
    Размеры до 1b (0.77b для large)
    
    OCR ноут - https://www.kaggle.com/code/ademboukhris/florence-2-large-ocr-images-real-life-scenarios
    Микро статья на roboflow по OCR на Florence2 - https://blog.roboflow.com/florence-2-ocr/
    Пример прогнать - https://huggingface.co/spaces/gokaygokay/Florence-2
    
## SLMm
- Sage (MIT) 2023 [2]
  Links
  - github - https://github.com/ai-forever/sage
  - huggingface - https://huggingface.co/collections/ai-forever/sage-v110-release-660abac12d0769b9c67be501
  
  Фреймворк от Cбера для спеллчека модельки одни из лучших для задачи
  Хабр статья - https://habr.com/ru/companies/sberdevices/articles/763932/
  колаб демка - https://colab.research.google.com/github/ai-forever/sage/blob/main/notebooks/text_correction_demo.ipynb

## other
### статьи
#### major
- https://habr.com/ru/companies/sberbank/articles/716796/ - статья Сбера про рукописную детекцию задачи. [1]
  хаба статьи - https://github.com/ai-forever/ReadingPipeline
  Инсайты:
    - возможно генерировать синту через GAN (https://habr.com/ru/companies/sberbank/articles/589537/)
- https://habr.com/ru/articles/815727/ - построение аналогичного сервиса с использованием Яндекс OCR
- https://habr.com/ru/companies/sberdevices/articles/763932/ - статья Сбера про коррекцию ошибок с помощью SLM [2]
- https://habr.com/ru/companies/jetinfosystems/articles/660405/ - статья про сравнение EasyOCR, Pytesseract, PaddleOCR
  есть классное сравнение фреймворков, из которого можно сделать вывод, что EasyOCR является лучшим выбором для быстрого бейзлайна
  есть идеи по препроцессингу, к примеру про выравнивание по сегментации. В конечном итоге использовали scikit-image и descew
  есть классная деталь про то что EasyOCR делает в базе 4 детекции (0, 90, 180, 270 градусов поворота)[3]
#### minor
- https://habr.com/ru/articles/533350/ - статья похожая на ту что выше
- https://habr.com/ru/articles/895664/ - слабая статья про сравнение LLM для задачи, лишний раз подтвердил нужность именно Qwen vl(72b)
- https://habr.com/ru/articles/720614/ - относительно новая сатья про построение детектора HTR (eng)
  github - https://github.com/CyberLympha/Examples/blob/main/CV/CRNN_for_IAM.ipynb 
- https://habr.com/ru/companies/mws/articles/832504/ - есть небольшая вырезка про surya
- https://habr.com/ru/companies/yandex/articles/712510/ - статья о Yandex OCR для архивных документов, есть идеи для базового пайплайна
  Слабо применимо из за научпоп формата, но общие идеи про архитектуру или разбивку текста на строки-сегменты есть
- https://habr.com/ru/articles/691598/ - статья про дообучение EasyOCR на русский язык
- https://habr.com/ru/articles/573030/ - статья сравнение Pytesseract, FineReader, EasyOCR[4]
