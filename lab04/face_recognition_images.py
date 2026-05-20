import cv2
import pickle
import os
from utils import get_face_locations, get_face_encodings, nb_of_matches

def recognize_faces(test_image_path, pickle_path="encodings.pickle"):
    print("[Інфо] Завантаження збереженої бази кодувань облич...")
    if not os.path.exists(pickle_path):
        print(f"[Помилка] Файл '{pickle_path}' не знайдено! Спочатку запустіть face_encoding.py.")
        return

    with open(pickle_path, "rb") as f:
        known_encodings = pickle.load(f)

    print(f"[ІНФО] Аналіз тестового зображення '{test_image_path}'...")
    if not os.path.exists(test_image_path):
        print(f"[Помилка] Тестове фото '{test_image_path}' не знайдено.")
        return

    # Пошук облич та отримання вкладень через utils.py
    locations, rgb_image = get_face_locations(test_image_path)
    encodings = get_face_encodings(rgb_image, locations)

    # Для графічного відображення OpenCV використовує BGR формат
    bgr_output_image = cv2.imread(test_image_path)

    # Ітерація по кожному знайденому обличчю на тестовому кадрі
    for (top, right, bottom, left), test_encoding in zip(locations, encodings):
        counts = {}

        # Цикл пошуку обличчя у базі (як показано на Рис. 20 посібника)
        for person_name, person_encodings in known_encodings.items():
            # Отримуємо кількість збігів за допомогою нашої функції голосування
            matches_count = nb_of_matches(person_encodings, test_encoding)
            counts[person_name] = matches_count

        # Визначаємо ім'я особи з максимальною кількістю збігів
        name = "Unknown"
        if counts and max(counts.values()) > 0:
            name = max(counts, key=counts.get)

        # Малювання обмежувальної рамки (Bounding Box) засобами OpenCV
        cv2.rectangle(bgr_output_image, (left, top), (right, bottom), (0, 255, 0), 2)
        
        # Додавання фонової плашки для тексту під рамкою
        cv2.rectangle(bgr_output_image, (left, bottom - 25), (right, bottom), (0, 255, 0), cv2.FILLED)
        cv2.putText(bgr_output_image, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)

    # Вивід вікна з фінальним результатом розпізнавання
    cv2.imshow("Face Recognition System", bgr_output_image)
    print("\n--- Вікно розпізнавання активне. Натисніть будь-яку клавішу для закриття програми ---")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Вкажіть ім'я або повний шлях до зображення, яке хочете протестувати
    recognize_faces("test_image.jpg")
