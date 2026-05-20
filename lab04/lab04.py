import os
import sys
import cv2

# Додаємо поточну директорію до шляху пошуку модулів, 
# щоб Python гарантовано бачив файли utils.py, face_encoding.py та face_recognition_images.py
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Вказуємо шлях до зображень
ABSOLUTE_PATH = os.getcwd()
LAB_PATH = ABSOLUTE_PATH + "/lab04/"

DATASET_DIR = LAB_PATH + "/dataset"
PICKLE_FILE = LAB_PATH + "/encodings.pickle"
TEST_IMAGE = LAB_PATH + "/the_expendables_2.jpg"

try:
    from face_encoding import encode_dataset
    from face_recognition_images import recognize_faces
except ImportError as e:
    print("[ПОМИЛКА] Не вдалося імпортувати необхідні модулі.")
    print("Переконайтеся, що файли 'face_encoding.py', 'face_recognition_images.py' та 'utils.py' "
          "знаходяться в тій самій папці, що й цей скрипт.")
    print(f"Деталі помилки: {e}")
    sys.exit(1)

def main():
    print("=" * 70)
    print("  Головний керуючий сценарій для лабораторної роботи №4")
    print("  Тема: Розпізнавання людини на фото з використанням бібліотеки Dlib")
    print("=" * 70)

    # --- ЕТАП 1: Валідація структури папок перед початком ---
    print("\n[КРОК 1/3] Перевірка наявності вхідних даних...")
    if not os.path.exists(DATASET_DIR):
        print(f"[УВАГА] Папку еталонів '{DATASET_DIR}' не знайдено.")
        print(f"Створюємо порожню папку '{DATASET_DIR}'. Будь ласка, додайте туди підпапки з фото відомих людей.")
        os.makedirs(DATASET_DIR)
        sys.exit(0)
    else:
        # Перевіряємо, чи є всередині папки з обличчями
        subdirs = [d for d in os.listdir(DATASET_DIR) if os.path.isdir(os.path.join(DATASET_DIR, d))]
        if not subdirs:
            print(f"[Попередження] Папка '{DATASET_DIR}' порожня! Створіть у ній підпапки (наприклад, '{DATASET_DIR}/Ivan/')")
            print("і покладіть туди фотографії облич для навчання.")
            sys.exit(0)
        print(f"[Успіх] Знайдено осіб для кодування: {len(subdirs)} ({', '.join(subdirs)})")

    # --- ЕТАП 2: Генерація або оновлення бази кодувань ---
    print("\n[КРОК 2/3] Запуск процесу сканування та генерації цифрових відбитків облич...")
    print("-" * 70)
    # Викликаємо функцію генерації з face_encoding.py
    encode_dataset(dataset_path=DATASET_DIR, output_pickle=PICKLE_FILE)
    print("-" * 70)

    # Подвійна перевірка, чи з'явився файл після роботи скрипта
    if not os.path.exists(PICKLE_FILE):
        print(f"[ПОМИЛКА] Файл '{PICKLE_FILE}' не був створений. Перевірте помилки у модулі face_encoding.py.")
        sys.exit(1)
    print(f"[УСПІХ] База даних успішно серіалізована у файл: {PICKLE_FILE}")

    # --- ЕТАП 3: Розпізнавання облич на тестовому зображенні ---
    print("\n[КРОК 3/3] Налаштування тестового зображення для ідентифікації...")
    if not os.path.exists(TEST_IMAGE):
        print(f"[ПОМИЛКА] Тестове зображення '{TEST_IMAGE}' не знайдено у корені проекту.")
        print("Будь ласка, покладіть туди будь-яке фото, назвіть його 'test_image.jpg' і запустіть скрипт знову.")
        sys.exit(1)

    print(f"[ІНФО] Запуск розпізнавання для файлу: '{TEST_IMAGE}'")
    print("Очікуйте, відкривається графічне вікно результату...")
    print("-" * 70)
    
    # Викликаємо функцію ідентифікації з face_recognition_images.py
    recognize_faces(test_image_path=TEST_IMAGE, pickle_path=PICKLE_FILE)
    
    print("-" * 70)
    print("[ІНФО] Роботу головного оркестраційного скрипта лабораторної роботи №4' успішно завершено!")
    print("=" * 70)

if __name__ == "__main__":
    main()