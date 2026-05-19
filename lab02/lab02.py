import cv2
import os

def wait_and_clear(task_num : float, next_task : str = "продовжити"):
    print(f"\n--- Натисніть ENTER (в консолі або на вікні зображення) щоб закрити завдання {task_num} і {next_task}) ---")
    cv2.waitKey(1)
    input()
    cv2.destroyAllWindows()

# 1. --- Перевірка версії OpenCV ---
print("OpenCV Version:", cv2.__version__)
wait_and_clear(1)

# Вказуємо шлях до зображень
absolute_path = os.getcwd()
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
    
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 1. Виявлення облич
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    print(f"Знайдено осіб на фото: {len(faces)}")
    
    for (x, y, w, h) in faces:
        # 1. Виявлення облич (синій колір)
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        
        # 2. Виявлення очей в межах обличчя (жовтий колір)
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 255), 2)
            
        # 3. Виявлення посмішки (червоний колір)
        smiles = smile_cascade.detectMultiScale(roi_gray, 1.8, 20)
        for (sx, sy, sw, sh) in smiles:
            cv2.rectangle(roi_color, (sx, sy), (sx+sw, sy+sh), (0, 0, 255), 2)

    cv2.imshow('Detection Results', img)
    wait_and_clear(2)

def process_video(source=0):
    """source=0 - веб-камера, або шлях до відеофайлу"""
    cap = cv2.VideoCapture(source)
    print("Натисніть 'q' для виходу з відео.")
    
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Детекція облич
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
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
            smiles = smile_cascade.detectMultiScale(roi_gray, 1.8, 20)
            for (sx, sy, sw, sh) in smiles:
                cv2.rectangle(roi_color, (sx, sy), (sx+sw, sy+sh), (0, 0, 255), 2)
            
        # Детекція пішоходів (HOG або каскади, тут каскади для прикладу)
        bodies = body_cascade.detectMultiScale(gray, 1.1, 3)
        for (x, y, w, h) in bodies:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 165, 255), 4)
            
        cv2.imshow('Video Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

# Виконання
process_photo(image_path)
process_video(video1_path)
process_video(video2_path)
