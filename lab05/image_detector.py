import os
from imageai.Detection import ObjectDetection

# ==============================================================================
# КОНФІГУРАЦІЯ: Перемикач версії ImageAI та бекенду
# True  - Використовувати старі моделі Keras/TensorFlow (.h5) (ImageAI v2.x)
# False - Використовувати нові моделі PyTorch (.pt) (ImageAI v3.0+)
# ==============================================================================
USE_LEGACY_H5 = False

ABSOLUTE_PATH = os.getcwd()
LAB_PATH = ABSOLUTE_PATH + "/lab05/"

H5_MODEL_PATH = LAB_PATH + "/yolo-tiny.h5"
PT_MODEL_PATH = LAB_PATH + "/tiny-yolov3.pt"
INPUT_PATH = LAB_PATH + "/fruits.jpg"
OUTPUT_PATH = LAB_PATH + "/output/newimage.jpg"

def run_image_detection():
    print("[IMAGE] Початок обробки зображення...")
    
    print(INPUT_PATH + "/n")
    print(OUTPUT_PATH + "/n")

    # Вибір моделі залежно від конфігурації
    if USE_LEGACY_H5:
        model_path = H5_MODEL_PATH
    else:
        # Новий формат моделей для PyTorch
        model_path = PT_MODEL_PATH 

    # Ініціалізація стандартного детектора
    detector = ObjectDetection()
    
    # Щоб використовувати власноруч навчену модель (Custom),
    # відповідно до документації ImageAI v3, розкоментуйте наступні 3 рядки 
    # та закоментуйте стандартний ObjectDetection() вище:
    # from imageai.Detection.Custom import CustomObjectDetection
    # detector = CustomObjectDetection()
    # detector.setJsonPath("./models/detection_config.json") # Custom потребує JSON словник

    detector.setModelTypeAsTinyYOLOv3()
    detector.setModelPath(model_path)
    detector.loadModel()

    # Виявлення об'єктів
    detection = detector.detectObjectsFromImage(
        input_image=INPUT_PATH,
        output_image_path=OUTPUT_PATH,
        minimum_percentage_probability=30
    )
    
    print("\n--- Знайдені об'єкти на фото ---")
    for eachItem in detection:
        print(f"{eachItem['name']}: {eachItem['percentage_probability']}%")
        
    print("[IMAGE] Обробку завершено. Результат збережено у папці output.")