import image_detector
import video_detector
import os

ABSOLUTE_PATH = os.getcwd()
LAB_PATH = ABSOLUTE_PATH + "/lab05/"

def wait_for_user(step_name):
    print(f"\n[ПАУЗА] Завершено етап: {step_name}")
    input("Натисніть ENTER, щоб продовжити до наступного кроку...")
    print("-" * 60)

def main():
    print("=" * 60)
    print("  АВТОМАТИЗОВАНИЙ ЗАПУСК ЛАБОРАТОРНОЇ РОБОТИ №5")
    print("=" * 60)
    
    # Перевірка наявності потрібної моделі
    model_name = "tiny-yolov3.pt" if not image_detector.USE_LEGACY_H5 else "yolo-tiny.h5"
    model_path = f"{LAB_PATH}/{model_name}"

    if not os.path.exists(model_path):
        print(f"[ПОМИЛКА] Модель {model_name} не знайдено у папці {LAB_PATH}")
        return

    # --- КРОК 1: Обробка зображення ---
    print("\n[КРОК 1/2] Запуск розпізнавання на фото...")
    try:
        image_detector.run_image_detection()
        wait_for_user("Розпізнавання на фото")
    except Exception as e:
        print(f"Помилка на кроці 1: {e}")
        return

    # --- КРОК 2: Обробка відео ---
    print("\n[КРОК 2/2] Запуск розпізнавання на відео...")
    try:
        video_detector.run_video_detection()
        print("[УСПІХ] Обробку відео завершено.")
    except Exception as e:
        print(f"Помилка на кроці 2: {e}")
        return

    print("=" * 60)
    print("Усі етапи лабораторної роботи виконано успішно!")
    print("=" * 60)

if __name__ == "__main__":
    # Гарантуємо наявність необхідних директорій
    for directory in ["output"]:
        if not os.path.exists(LAB_PATH + directory):
            os.makedirs(LAB_PATH + directory)
    main()