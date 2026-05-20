import cv2
import os
import dlib
# import face_recognition
import numpy as np

# ==============================================================================
# КОНФІГУРАЦІЯ: Перемикач архітектури розпізнавання
# True  - Використовувати прямі низькорівневі виклики бібліотеки Dlib (Рис. 7-12)
# False - Використовувати стандартну обгортку бібліотеки face_recognition
# ==============================================================================
USE_DLIB_DIRECT = True

# Вказуємо шлях до зображень
ABSOLUTE_PATH = os.getcwd()
DLIB_DATA_PATH = ABSOLUTE_PATH + "/lab04/dlib_data/"

# Ініціалізація компонентів Dlib (завантажуються один раз при імпорті модуля)
if USE_DLIB_DIRECT:
    print("[UTILS] Налаштовано режим: Прямий конвеєр Dlib (Низький рівень)")
    dlib_detector = dlib.get_frontal_face_detector()
    
    # Шляхи до стандартних ваг моделей Dlib з Рис. 7 та Рис. 10 методички
    # Переконайтеся, що ці файли розпаковані та лежать у папці вашого проекту
    try:
        dlib_predictor = dlib.shape_predictor(DLIB_DATA_PATH + "/shape_predictor_68_face_landmarks.dat")
        dlib_facerec = dlib.face_recognition_model_v1(DLIB_DATA_PATH + "/dlib_face_recognition_resnet_model_v1.dat")
    except RuntimeError:
        print("[ПОПЕРЕДЖЕННЯ] Не знайдено файли моделей '.dat' у папці проекту!")
        print("Будь ласка, завантажте 'shape_predictor_68_face_landmarks.dat' та 'dlib_face_recognition_resnet_model_v1.dat'")
else:
    print("[UTILS] Налаштовано режим: Високорівнева обгортка face_recognition")

def get_face_locations(image_path, model="hog"):
    """
    Завантажує зображення та знаходить координати (локації) всіх облич на ньому.
    Повертає уніфікований формат локацій: [(top, right, bottom, left), ...] та RGB зображення.
    """
    if USE_DLIB_DIRECT:
        # Dlib працює як з колірними просторами, так і з градаціями сірого.
        # Згідно з Рис. 8 завантажуємо через OpenCV та переводимо в RGB
        bgr_img = cv2.imread(image_path)
        if bgr_img is None:
            raise FileNotFoundError(f"Не вдалося завантажити зображення: {image_path}")
        rgb_image = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
        
        # Детекція за допомогою фронтального детектора Dlib (Рис. 8)
        # Параметр 1 означає апсемплінг зображення 1 раз для пошуку дрібніших облич
        dets = dlib_detector(rgb_image, 1)
        
        # Конвертуємо об'єкти dlib.rectangle в уніфікований формат (top, right, bottom, left)
        # щоб не ламати логіку малювання в інших файлах лабораторної
        locations = []
        for det in dets:
            locations.append((det.top(), det.right(), det.bottom(), det.left()))
        return locations, rgb_image
    # else:
        # Стандартний шлях через обгортку face_recognition
        # rgb_image = face_recognition.load_image_file(image_path)
        # locations = face_recognition.face_locations(rgb_image, model=model)
        # return locations, rgb_image


def get_face_encodings(image, face_locations):
    """
    Генерує 128-вимірні вектори ознак (вкладення) для кожної знайденої області обличчя.
    """
    if USE_DLIB_DIRECT:
        encodings = []
        for (top, right, bottom, left) in face_locations:
            # Конвертуємо координати назад у dlib.rectangle для пошуку точок
            rect = dlib.rectangle(left, top, right, bottom)
            
            # Рис. 9: Визначення 68 антропометричних точок обличчя
            shape = dlib_predictor(image, rect)
            
            # Рис. 11-12: Обчислення 128-вимірного дескриптора (вектора ознак)
            face_descriptor = dlib_facerec.compute_face_descriptor(image, shape)
            
            # Конвертуємо об'єкт дескриптора Dlib у звичайний масив NumPy
            encoding_np = np.array(face_descriptor)
            encodings.append(encoding_np)
        return encodings
    # else:
    #    # Стандартний шлях через обгортку face_recognition
    #    return face_recognition.face_encodings(image, face_locations)


def nb_of_matches(known_person_encodings, test_encoding, tolerance=0.6):
    """
    Обчислює кількість збігів між одним тестовим вкладенням 
    та списком усіх відомих вкладень конкретної особи з бази даних.
    """
    if USE_DLIB_DIRECT:
        # --- Модифікований чистий математичний підхід за допомогою NumPy ---
        match_count = 0
        for known_encoding in known_person_encodings:
            # Обчислюємо Евклідову відстань між двома 128-вимірними векторами:
            # Спершу віднімаємо вектори, а потім знаходимо L2-норму за допомогою np.linalg.norm
            distance = np.linalg.norm(np.array(known_encoding) - np.array(test_encoding))
            
            # Якщо відстань менша за поріг схожості (tolerance), фіксуємо збіг
            if distance <= tolerance:
                match_count += 1
        return match_count
    #else:
    #    # --- Стандартний метод через бібліотеку face_recognition ---
    #    matches = face_recognition.compare_faces(known_person_encodings, test_encoding, tolerance=tolerance)
    #    return sum(matches)