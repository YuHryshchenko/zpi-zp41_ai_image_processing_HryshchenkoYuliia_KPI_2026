import os
import pickle
from utils import get_face_locations, get_face_encodings

def encode_dataset(dataset_path="dataset", output_pickle="encodings.pickle"):
    print("[ІНФО] Початок генерації вкладень облич із набору даних...")
    known_encodings = {}

    if not os.path.exists(dataset_path):
        print(f"[ПОМИЛКА] Директорію '{dataset_path}' не знайдено! Створіть її.")
        return

    # Перебираємо папки всередині директорії dataset (назва папки = ім'я людини)
    for person_name in os.listdir(dataset_path):
        person_dir = os.path.join(dataset_path, person_name)
        
        # Перевіряємо, чи це дійсно папка особи
        if not os.path.isdir(person_dir):
            continue
            
        print(f"[ІНФО] Обробка фотографій для: {person_name}")
        if person_name not in known_encodings:
            known_encodings[person_name] = []

        # Перебираємо всі файли зображень у папці людини
        for image_name in os.listdir(person_dir):
            if not image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue
                
            image_path = os.path.join(person_dir, image_name)
            
            try:
                # Використовуємо функції модулю utils
                locations, image = get_face_locations(image_path)
                encodings = get_face_encodings(image, locations)
                
                # Додаємо знайдені кодування до списку поточної особи
                for encoding in encodings:
                    known_encodings[person_name].append(encoding)
                    
            except Exception as e:
                print(f"[ПОМИЛКА] Не вдалося обробити файл {image_path}: {e}")

    # Збереження словника вкладень на диск за допомогою модуля pickle
    print(f"[ІНФО] Збереження згенерованих вкладень у файл '{output_pickle}'...")
    with open(output_pickle, "wb") as f:
        pickle.dump(known_encodings, f)
        
    print("[УСПІХ] Процес завершено. Файл 'encodings.pickle' успішно створено!")

if __name__ == "__main__":
    encode_dataset()
