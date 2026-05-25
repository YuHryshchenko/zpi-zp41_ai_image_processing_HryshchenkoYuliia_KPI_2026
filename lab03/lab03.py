import os
import cv2
import numpy as np

# Вказуємо шлях до зображень
absolute_path = os.getcwd() # Отримати поточну робочу директорію (папку), в якій зараз виконується Python-скрипт
image_path = absolute_path + '/lab03/lane_markings.jpg'
video_path = absolute_path + '/lab03/854669-hd_1920_1080_30fps.mp4'

def wait_and_clear(task_num : float, next_task : str = "продовжити"):
    print(f"\n--- Натисніть ENTER (в консолі або на вікні зображення) щоб закрити завдання {task_num} і {next_task}) ---")
    key = cv2.waitKey(0) & 0xFF
    if key == 13:
      print(f"\n--- {next_task} ---")
    cv2.destroyAllWindows()

def draw_lines(img, lines, color=[255, 0, 0], thickness=7): # img -> Зображення дороги, lines -> Набір знайдених коротких ліній, color=[255,0,0] -> це синій колір, thickness=7 -> Товщина лінії = 7 px
    """Calculates average slopes and draws unified left and right lane lines.""" # Обчислює середні ухили та малює об'єднані лінії лівої та правої смуг руху.
    x_bottom_pos, x_upper_pos = [], [] # права смуга
    x_bottom_neg, x_upper_neg = [], [] # ліва смуга

    # y_bottom = 540 # Bottom of the image (adjust based on your image size) # Нижня частина зображення (налаштуйте розмір відповідно до вашого зображення)
    # y_upper = 315  # Top of the ROI (adjust based on your image size) # Верхня частина ROI (налаштуйте відповідно до розміру зображення)
    # Отже лінії малюються від низу дороги до середини дороги
    y_bottom = img.shape[0] # Визначаються межі малювання: Низ картинки
    y_upper = int(img.shape[0] * 0.5) # Визначаються межі малювання: Половина картинки у напрямку вверх

    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                # Prevent division by zero # Запобігання діленню на нуль
                if x2 - x1 == 0: 
                    continue
                
                slope = (y2 - y1) / (x2 - x1) # Розрахунок нахилу
                
                # Test and filter values based on slope to separate left/right lanes
                if 0.5 < slope < 0.8: # Якщо slope > 0 -> Лінія нахилена вправо
                    b = y1 - slope * x1 # Обчислення перетину
                    x_bottom_pos.append((y_bottom - b) / slope) # Знаходимо X внизу -> (y_bottom - b) / slope (де ця лінія перетне нижню межу), append - додати, intercept - перехоплення
                    x_upper_pos.append((y_upper - b) / slope) # Знаходимо X зверху -> (y_upper - b) / slope (де ця лінія перетне верхню межу)
                elif -0.8 < slope < -0.5: # Якщо slope < 0 -> Лінія нахилена вліво
                    b = y1 - slope * x1
                    x_bottom_neg.append((y_bottom - b) / slope)
                    x_upper_neg.append((y_upper - b) / slope)

    # Draw Positive (Right) Line # Малюємо усереднену праву смугу
    if len(x_bottom_pos) > 0 and len(x_upper_pos) > 0:
        cv2.line(img, 
                 (int(np.mean(x_bottom_pos)), int(y_bottom)), # Беремо середнє -> Це робить одну стабільну красиву лінію замість шумних шматків
                 (int(np.mean(x_upper_pos)), int(y_upper)), 
                 color, thickness)

    # Draw Negative (Left) Line # Малюємо усереднену ліву смугу
    if len(x_bottom_neg) > 0 and len(x_upper_neg) > 0:
        cv2.line(img, 
                 (int(np.mean(x_bottom_neg)), int(y_bottom)), 
                 (int(np.mean(x_upper_neg)), int(y_upper)), 
                 color, thickness)

def process_image(img_path):
    """Processes a single image step-by-step for demonstration.""" # Покроково обробляє одне зображення для демонстрації
    print("Starting Step-by-Step Image Processing...") # Повідомлення про старт
    
    # 1. Load original image
    img = cv2.imread(img_path) # Читання файлу
    if img is None:
        print(f"Error: Could not load image at {img_path}")
        return
    cv2.imshow('1 - Input image', img) # Відобразити оригіналу зображення
    wait_and_clear(1)

    # 2. Convert to Grayscale
    grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # Перетворення зображення в відтінки сірого
    cv2.imshow('2 - Grayscale', grayscale) # Відобразити зображення у відтінках сірого
    wait_and_clear(2)

    # 3. Apply Gaussian Blur # Розмиття прибирає шум
    blur_ksize = 9 # Досить сильне згладжування, щоб не було реакції ліній на дрібні деталі
    blur = cv2.GaussianBlur(grayscale, (blur_ksize, blur_ksize), 0)
    cv2.imshow('3 - Blurred', blur) # Відобразити розмите зображення
    wait_and_clear(3)

    # 4. Canny Edge Detection # Виявлення країв
    low_t = 50 # Параметри: 50 нижній поріг
    high_t = 150 # Параметри: 150 верхній поріг
    edges = cv2.Canny(blur, low_t, high_t) # Результат: білі контури на чорному фоні.
    cv2.imshow('4 - Canny Edges', edges) # Відобразити чорно-біле зображення
    wait_and_clear(4)

    # 4.1 Mask out car hood and the dust at the bottom # Маскуємо капот автомобіля та пил знизу
    h, w = edges.shape[:2]
    # Area: from 0% width and 83% height to the bottom-right corner
    cv2.rectangle(edges, (0, int(h * 0.83)), (w, h), 0, -1) # cv2.rectangle(edges, ..., 0, -1) -> Видаляємо капот машини
    cv2.imshow('4.1 - Edges without Hood', edges) # Відобразити чорно-біле зображення без капоту
    wait_and_clear(4.1)

    # 4.2 Dilate edges to make lines thicker # Розширте краї, щоб зробити лінії товстішими 
    dilate_ksize = 20
    kernel_dilate = np.ones((dilate_ksize, dilate_ksize), np.uint8) # Створюємо ядро
    dilated_edges = cv2.dilate(edges, kernel_dilate, iterations=1) # Розширюємо контури
    cv2.imshow('4.2 - Dilated Edges', dilated_edges) # Відобразити чорно-біле зображення без капоту з розширеними контурами
    wait_and_clear(4.2)

    # 5. Region of Interest (ROI) Mask
    # h, w already obtained from edges.shape above
    # Using a hexagon ROI to capture the full width at the bottom and sides and taper toward the horizon to focus on the lanes.
    # Використання шестикутної області інтересу для захоплення повної ширини внизу та з боків та звуження до горизонту, щоб зосередитися на доріжках.
    vertices = np.array([    # Це область дороги
        [
            (int(w * 0.3), h),               # bottom-left
            (w, h),                          # bottom-right
            (w, int(h * 0.7)),               # mid-right (30% up from bottom)
            (int(w * 0.66), int(h * 0.42)),  # top-right
            (int(w * 0.62), int(h * 0.42)),  # top-left
            (int(w * 0.3), int(h * 0.8))     # mid-left (80% up from bottom)
        ]
    ], dtype=np.int32)

    mask = np.zeros_like(edges) # Створюємо маску
    ignore_mask_color = 255
    masked_poly = cv2.fillPoly(mask, vertices, ignore_mask_color) # Заповнюємо ROI білим кольором
    cv2.imshow('5 - Masked Poly', masked_poly) # Відобразити зображення
    wait_and_clear(5.1)

    masked_edges = cv2.bitwise_and(dilated_edges, mask) # Apply mask to dilated edges # Застосувати маску на розширені краї
    cv2.imshow('5 - Masked Edges', masked_edges) # Відобразити зображення
    wait_and_clear(5.2)

    # 6. Hough Transform 
    rho = 3 # точність по відстані
    theta = np.pi / 180
    threshold = 15
    min_line_len = 150 # короткі лінії ігноруються
    max_line_gap = 60 # дозволяє з’єднувати розриви
    
    lines = cv2.HoughLinesP(masked_edges, rho, theta, threshold, np.array([]), 
                            minLineLength=min_line_len, maxLineGap=max_line_gap) # Знаходить прямі лінії
    
    # 7. Draw Lines on the original image
    result_img = img.copy() # Створюємо копію картинки
    draw_lines(result_img, lines) # Малюємо усереднені смуги
    cv2.imshow('6 - Final Result with Hough Lines', result_img) # Відобразити зображення -> Показує готовий результат
    wait_and_clear(6)

def process_video(video_path):
    """Applies the pipeline to a video stream."""
    print("Starting Video Processing... Press 'q' to stop.") # Повідомлення про старт
    video_capture = cv2.VideoCapture(video_path) # Відкриваємо відео
    
    while video_capture.isOpened():
        ret, frame = video_capture.read() # ret=True якщо вдалося прочитати, frame = поточний кадр
        if ret:
            # Inline processing for the video frame # Вбудована обробка відеокадру
            blur_ksize = 9
            grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # Прибираємо колір
            blur = cv2.GaussianBlur(grayscale, (blur_ksize, blur_ksize), 0) # Прибирає шум, робимо розмиття
            edges = cv2.Canny(blur, 50, 150) # Шукаємо межі
            
            h, w = edges.shape[:2] # # Маскуємо капот автомобіля та пил знизу
            # Mask out car hood to prevent it from interfering with Hough Transform
            cv2.rectangle(edges, (0, int(h * 0.83)), (w, h), 0, -1) # cv2.rectangle(edges, ..., 0, -1) -> Видаляємо капот машини

            # Dilate edges to make lines thicker for better Hough Transform detection # Розширте краї, щоб зробити лінії товстішими
            dilate_ksize = 20
            kernel_dilate = np.ones((dilate_ksize, dilate_ksize), np.uint8) # Створюємо ядро
            dilated_edges = cv2.dilate(edges, kernel_dilate, iterations=1) # Розширюємо контури

            vertices = np.array([ # Це область дороги
                [
                    (int(w * 0.3), h),               # bottom-left
                    (w, h),                          # bottom-right
                    (w, int(h * 0.7)),               # mid-right (30% up from bottom)
                    (int(w * 0.66), int(h * 0.42)),  # top-right
                    (int(w * 0.62), int(h * 0.42)),  # top-left
                    (int(w * 0.3), int(h * 0.8))     # mid-left (80% up from bottom)
                ]
            ], dtype=np.int32)

            mask = np.zeros_like(edges) # Створюємо маску
            cv2.fillPoly(mask, vertices, 255) # Fill the mask with white # Заповнюємо ROI білим кольором
            masked_edges = cv2.bitwise_and(dilated_edges, mask) # Apply mask to dilated edges # Застосувати маску на розширені краї
            
            lines = cv2.HoughLinesP(masked_edges, 3, np.pi / 180, 15, np.array([]), 
                                    minLineLength=150, maxLineGap=60) # Знаходить прямі лінії
            
            draw_lines(frame, lines) # Малюємо усереднені смуги
            
            cv2.imshow('Video Frame', frame) # Відображаємо відео
            
            if cv2.waitKey(1) & 0xFF == ord('q'): # waitKey(1) -> Щосекунди OpenCV слухає клавіатуру
                break
        else:
            break
            
    video_capture.release() # Звільняємо файл / камеру
    cv2.destroyAllWindows()

# Виконання
process_image(image_path)
process_video(video_path)
