import cv2
import os

def wait_and_clear(task_num : float, next_task : str = "продовжити"):
    print(f"\n--- Натисніть ENTER (в консолі або на вікні зображення) щоб закрити завдання {task_num} і {next_task}) ---")
    key = cv2.waitKey(0) & 0xFF
    if key == 13:
      print(f"\n--- {next_task} ---")
    cv2.destroyAllWindows()

# 1. --- Перевірка версії OpenCV ---
print("OpenCV Version:", cv2.__version__)
wait_and_clear(1)

# Вказуємо шлях до зображень
absolute_path = os.getcwd() # Отримати поточну робочу директорію (папку), в якій зараз виконується Python-скрипт
image_path = absolute_path + '/lab02/petapixel6.jpg'
video1_path = absolute_path + '/lab02/39837-424360872_medium.mp4'
video2_path = absolute_path + '/lab02/2121-155244120_medium.mp4'

# Шляхи до XML файлів
face_cascade = cv2.CascadeClassifier(absolute_path + '/lab02/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(absolute_path + '/lab02/haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier(absolute_path + '/lab02/haarcascade_smile.xml')
body_cascade = cv2.CascadeClassifier(absolute_path + '/lab02/haarcascade_fullbody.xml')

def process_photo(image_path):
    if not os.path.exists(image_path):
        print(f"Файл {image_path} не знайдено.")
        return
    
    img = cv2.imread(image_path) # Читання файлу
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # Перетворення зображення в відтінки сірого
    # cv2.imwrite(absolute_path + '/lab02/saved_foto_02.jpg', gray)
    
    # 1. Виявлення облич
    # img = cv2.imread(absolute_path + '/lab02/saved_foto_02.jpg')
    faces = face_cascade.detectMultiScale(gray, 1.3, 5) # 1.3 -> точний масштабний пошук, 5 -> середня строгість
    print(f"Знайдено осіб на фото: {len(faces)}") # Знайдено осіб на фото: 14
    
    for (x, y, w, h) in faces: # faces — це список облич, x, y → верхній лівий кут обличчя, w, h → ширина і висота
        # 1. Виявлення облич (синій колір)
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        
        # 2. Виявлення очей в межах обличчя (жовтий колір)
        eyes = eye_cascade.detectMultiScale(roi_gray) # Шукаємо очі лише всередині обличчя
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 255), 2)
            
        # 3. Виявлення посмішки (червоний колір)
        smiles = smile_cascade.detectMultiScale(roi_gray, 1.8, 20) # 1.8 -> великий крок масштабування (швидше, але грубіше), 20 -> дуже строгий фільтр (менше хибних спрацьовувань)
        for (sx, sy, sw, sh) in smiles:
            cv2.rectangle(roi_color, (sx, sy), (sx+sw, sy+sh), (0, 0, 255), 2)

    cv2.imshow('Detection Results', img) # Відобразити зображення
    wait_and_clear(2)

def process_video(source=0):
    """source=0 - веб-камера, або шлях до відеофайлу"""
    cap = cv2.VideoCapture(source) # Відкриття відеопотоку
    print("Натисніть 'q' для виходу з відео.")
    
    while True:
        ret, frame = cap.read() # cap.read() -> береться один кадр з камери; ret -> отриман чи не отриман кадр; frame -> сам кадр
        if not ret: break # Якщо кадру немає -> завершити
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # Перетворення зображення на відео у відтінки сірого
        
        # Детекція облич
        faces = face_cascade.detectMultiScale(gray, 1.1, 4) # 1.1 -> точний масштабний пошук, 4 -> середня строгість
        for (x, y, w, h) in faces:
            # 1. Виявлення облич (синій колір)
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

            # 2. Виявлення очей в межах обличчя (жовтий колір)
            eyes = eye_cascade.detectMultiScale(roi_gray)
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 255), 2)
                
            # 3. Виявлення посмішки (червоний колір)
            smiles = smile_cascade.detectMultiScale(roi_gray, 1.8, 20) # 1.8 -> великий крок масштабування (швидше, але грубіше), 20 -> дуже строгий фільтр (менше хибних спрацьовувань)
            for (sx, sy, sw, sh) in smiles:
                cv2.rectangle(roi_color, (sx, sy), (sx+sw, sy+sh), (0, 0, 255), 2)
            
        # Детекція пішоходів (HOG або каскади, тут каскади для прикладу)
        bodies = body_cascade.detectMultiScale(gray, 1.1, 3) # 1.1 -> точний масштабний пошук, 3 -> середня строгість
        for (x, y, w, h) in bodies:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 165, 255), 4) # (0,165,255) → помаранчевий колір
            
        cv2.imshow('Video Detection', frame) # Відобразити відео
        if cv2.waitKey(1) & 0xFF == ord('q'): # waitKey(1) -> Щосекунди OpenCV слухає клавіатуру
            break
            
    cap.release() # Це звільняє камеру
    cv2.destroyAllWindows()

# Виконання
process_photo(image_path)
process_video(video1_path)
process_video(video2_path)

# --- Відповіді на Контрольні запитання (Лабораторна робота №2) ---
print("Відповіді на Контрольні запитання (Лабораторна робота №2)\n")

print("1. Що таке алгоритм Віоли-Джонса?\n")
print("Це метод детекції об'єктів (переважно облич) у реальному часі, запропонований Полом Віолою та Майклом Джонсом у 2001 році." \
" Він базується на використанні ознак типу Хаара, інтегральних зображень для швидкого обчислення ознак та методу AdaBoost для відбору найважливіших ознак" \
" і створення каскаду класифікаторів.\n")

print("2. Що таке haarcascade?\n")
print("Це каскадний класифікатор на основі ознак Хаара. Це алгоритм машинного навчання, де каскадна функція тренується на великій кількості позитивних"
" (зображення з об'єктом) та негативних (зображення без об'єкта) прикладів. Він дозволяє ефективно відсіювати області зображення, де об'єкта точно немає," \
" значно прискорюючи процес пошуку.\n")

print("3. Що таке HOG-класифікатор?\n")
print("HOG (Histogram of Oriented Gradients — гістограма орієнтованих градієнтів) — це дескриптор ознак, що використовується в комп'ютерному зорі" \
" для виявлення об'єктів. Він аналізує розподіл інтенсивностей градієнтів (змін яскравості) у локальних областях зображення. Це дуже ефективно" \
" для виявлення об'єктів з чіткими формами (наприклад, людей, пішоходів).")

print("4. Що таке SVM-детектор?\n")
print("SVM (Support Vector Machine — метод опорних векторів) — це алгоритм машинного навчання, який використовується для класифікації даних." \
" У задачах комп'ютерного зору (наприклад, у поєднанні з HOG) він 'навчається' розрізняти вектор ознак об'єкта (наприклад, `людина` або `не людина`),"
" знаходячи оптимальну роздільну гіперплощину між класами.\n")

print("5. Що робить метод cvtColor та яка його мета використання у цій лабораторній?\n")
print("Метод cv2.cvtColor() перетворює зображення з одного колірного простору в інший (наприклад, з BGR у Grayscale). У цій лабораторній роботі" \
" його мета — перетворити кольорове зображення на чорно-біле (градації сірого), оскільки для роботи алгоритмів пошуку облич чи пішоходів колірна" \
" інформація є надлишковою і лише ускладнює обчислення. Робота з одним каналом (сірим) значно швидша і точніша для класифікаторів Хаара.\n")
