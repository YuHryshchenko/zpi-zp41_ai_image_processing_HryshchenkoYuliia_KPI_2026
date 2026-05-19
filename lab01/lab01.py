import os
import sys
import cv2
import imutils
import numpy as np

def wait_and_clear(task_num : float, next_task : str = "продовжити"):
    print(f"\n--- Натисніть ENTER (в консолі або на вікні зображення) щоб закрити завдання {task_num} і {next_task}) ---")
    cv2.waitKey(1)
    input()
    cv2.destroyAllWindows()

# 1. --- Перевірка версії OpenCV ---
print("OpenCV Version:", cv2.__version__)
wait_and_clear(1)

# Вказуємо шлях до зображення
absolute_path = os.getcwd()
image_path = absolute_path + '/lab01/cat.jpg'
image2_path = absolute_path + '/lab01/game_of_thrones.jpg'

# --- 2. Читання файлу (кольорове та у градаціях сірого) ---
img = cv2.imread(image_path)
cv2.imshow("Loaded - Original", img)
wait_and_clear(2.1)
img_gray = cv2.imread(image_path, 0) # 0 еквівалентно cv2.IMREAD_GRAYSCALE
cv2.imshow("Loaded - Grayscale", img_gray)
wait_and_clear(2.2)

if img is None:
    print(f"Помилка: Не вдалося завантажити зображення '{image_path}'.")
    sys.exit(1)

# 3. --- Відображення зображення у вікні ---
img = cv2.imread(image_path)
cv2.imshow("My image", img)
wait_and_clear(3)

# 4. --- Збереження файлу ---
cv2.imwrite(absolute_path + '/lab01/saved_foto.jpg', img_gray)
img = cv2.imread(absolute_path + '/lab01/saved_foto.jpg')
cv2.imshow("My gray image", img)
wait_and_clear(4)

# 5. --- Доступ до окремих пікселів (x=50, y=100) ---
# Порядок кольорів у OpenCV - BGR
img = cv2.imread(image_path)
if img.shape[0] > 100 and img.shape[1] > 50:
    (blue, green, red) = img[100, 50]
    print(f"Піксель на (50, 100) -> Червоний (R): {red}, Зелений (G): {green}, Синій (B): {blue}")
wait_and_clear(5)

# --- 6. Вирізання (Crop) регіону інтересу (ROI) ---
img = cv2.imread(image2_path)
if img.shape[0] > 160 and img.shape[1] > 420:
    roi = img[60:160, 320:420]
    cv2.imshow("ROI", roi)
wait_and_clear(6)

# 7. Зміна розміру зображення (непропорційна)
img = cv2.imread(image_path)
resized_square = cv2.resize(img, (200, 200))
cv2.imshow("Resized square", resized_square)
wait_and_clear(7)

# --- 8. Зміна розміру зображення (пропорційна, математичний підхід) ---
img = cv2.imread(image_path)
h, w = img.shape[0:2]
h_new = 300
ratio = w / h
w_new = int(h_new * ratio)
resized_prop = cv2.resize(img, (w_new, h_new))
cv2.imshow("Resized proportional", resized_prop)
wait_and_clear(8)

# --- 9. Зміна розміру за допомогою пакета imutils (зберігає пропорції автоматично) ---
img = cv2.imread(image_path)
resized_imutils = imutils.resize(img, width=300)
cv2.imshow("Resized imutils", resized_imutils)
wait_and_clear(9)

# 10. Поворот зображення за допомогою матриці поворотів (OpenCV)
h_rot, w_rot = resized_imutils.shape[0:2]
center = (w_rot // 2, h_rot // 2)
M = cv2.getRotationMatrix2D(center, -45, 1.0)
rotated_cv2 = cv2.warpAffine(resized_imutils, M, (w_rot, h_rot))
cv2.imshow("Rotated", rotated_cv2)
wait_and_clear(10)

# --- 11. Поворот зображення за допомогою imutils ---
img11 = cv2.imread(image_path)
rotated_imutils = imutils.rotate(resized_imutils, -45)
cv2.imshow("Rotated imutils", rotated_imutils)
wait_and_clear(11)

# --- 12. Розмиття зображення (Gaussian Blur) ---
img12 = cv2.imread(image_path)
blurred = cv2.GaussianBlur(resized_imutils, (11, 11), 0)
cv2.imshow("Blurred", blurred)
wait_and_clear(12)

# --- 13. Склеювання нормального та розмитого зображень ---
img13 = cv2.imread(image_path)
suming = np.hstack((resized_imutils, blurred))
cv2.imshow("Normal vs Blurred", suming)
wait_and_clear(13)

# --- 14. Малювання прямокутника на зображенні ---
img14 = cv2.imread(image_path)
img_rect = resized_imutils.copy()
cv2.rectangle(img_rect, (80, 170), (140, 220), (0, 0, 255), 2)
cv2.imshow("Rectangle", img_rect)
wait_and_clear(14)

# --- 15. Створення чорного зображення та малювання лінії ---
img_black = np.zeros((200, 200, 3), np.uint8)
cv2.line(img_black, (0, 0), (200, 200), (255, 0, 0), 5)
cv2.imshow("Line", img_black)
wait_and_clear(15)

# --- 16. Малювання полігону (ліній за набором точок) ---
img_black2 = np.zeros((200, 200, 3), np.uint8)
points = np.array([[0, 0], [100, 50], [50, 100], [0, 0]])
cv2.polylines(img_black2, np.int32([points]), True, (255, 255, 255))
cv2.imshow("Lines and Polygons", img_black2)
wait_and_clear(16)

# --- 17. Малювання кола ---
img_black3 = np.zeros((200, 200, 3), np.uint8)
cv2.circle(img_black3, (100, 100), 50, (0, 0, 255), 2)
cv2.imshow("Circle", img_black3)
wait_and_clear(17)

# --- 18. Розміщення тексту на зображенні ---
img_black4 = np.zeros((200, 550, 3), np.uint8)
font = cv2.FONT_HERSHEY_SCRIPT_COMPLEX
cv2.putText(img_black4, 'OpenCV', (0, 100), font, 4, (255, 255, 255), 4, cv2.LINE_4)
cv2.imshow("Text", img_black4)
wait_and_clear(18, "закінчити")

# --- Відповіді на Контрольні запитання (Лабораторна робота №1) ---
print("Відповіді на Контрольні запитання (Лабораторна робота №1)\n")

print("1) Що таке OpenCV?\n")
print("OpenCV (Open Source Computer Vision Library) — це бібліотека функцій та алгоритмів комп'ютерного зору, обробки зображень" \
" і чисельних алгоритмів загального призначення з відкритим кодом. Вона надає засоби для розпізнавання об'єктів"
" (наприклад, осіб, фігур, тексту), відстежування руху, перетворення зображень та застосування методів машинного навчання.\n")

print("2) Що таке піксель?\n")
print("Піксель - це найменший елемент зображення. Всі зображення утворюють прямокутну матрицю з пікселів. Кожен піксель зображення" \
" у форматі градації сірого має значення від 0 до 255, що відображає його інтенсивність. У кольоровому просторі"
" (наприклад, BGR у OpenCV) кожен піксель має додаткову інформацію, представлену 3-кортежем значень (синій, зелений, червоний)," \
" кожне з яких також знаходиться в діапазоні [0, 255].\n")

print("3) Як встановити пакет opencv-python?\n")
print("Встановити цей пакет можна за допомогою менеджера пакетів pip, виконавши у терміналі команду pip3 install opencv-python.\n")

print("4) Як розмити зображення?\n")
print("Для розмивання зображення та зменшення високочастотного шуму використовується функція Гаусового розмивання cv2.GaussianBlur()" \
" Наприклад: cv2.GaussianBlur(image, (11, 11), 0).\n")

print("5) Які функції використовуються для малювання та відображення тексту?\n")
print("Для малювання ліній: cv2.line()." \
"  Для малювання полігонів за точками: cv2.polylines()." \
"  Для малювання прямокутників: cv2.rectangle()." \
"  Для малювання кіл: cv2.circle()." \
"  Для відображення тексту на зображенні: cv2.putText().")