import os
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from PIL import Image

ABSOLUTE_PATH = os.getcwd()
LAB_PATH = ABSOLUTE_PATH + "/lab07/"

# Завантаження попередньо навченої моделі ESRGAN
model_url = "https://tfhub.dev/captain-pool/esrgan-tf2/1"
model = hub.load(model_url)

# ── Функція plot_image - вивід зображень ───────────────────────────────────
def plot_image(image, title=""): # Показує картинку
    image = np.asarray(image) # Перетворення у NumPy
    image = tf.clip_by_value(image, 0, 255) # Обмеження значень
    image = Image.fromarray(tf.cast(image, tf.uint8).numpy()) # Конвертація у uint8 -> unsigned integer 8-bit
    plt.imshow(image) # Показ через matplotlib
    plt.axis("off") # Прибираємо осі
    plt.title(title) # Ставимо заголовок

# ── Функція process_image - обробка зображень ──────────────────────────────
def process_image(directory_path, image_name):
    # ── Зчитування вхідного зображення ─────────────────────────────────────
    original_image = tf.image.decode_image(tf.io.read_file(directory_path + "/" + image_name)) # TensorFlow читає файл як tensor
    # Видаляємо alpha-канал, якщо формат вхідного зображення - PNG,
    # бо модель підтримає тільки зображення з 3 каналами
    if original_image.shape[-1] == 4:
        original_image = original_image[..., :-1] # Означає взяти всі канали, крім останнього

    # ── Підготовка зображення для моделі ───────────────────────────────────
    hr_size = (tf.convert_to_tensor(original_image.shape[:-1]) // 4) * 4 # Багато super-resolution моделей працюють лише з розмірами кратними 4
    original_image = tf.image.crop_to_bounding_box(original_image, 0, 0, hr_size[0], hr_size[1]) # Обрізає зайві пікселі
    # Модель очікує вхідне зображення в float32
    original_image = tf.cast(original_image, tf.float32) # Перетворення у float32
    # Модель очікує зображення з 4 вимірами(dimensions)
    original_image = tf.expand_dims(original_image, 0) # Додаємо batch dimension -> TensorFlow-модель очікує batch: (batch, height, width, channels)

    # ── Покращення зображення за допомогою моделі ──────────────────────────
    # Покращення зображення за допомогою моделі
    enhanced_image = model(original_image) # знаходить деталі, відновлює текстури, збільшує чіткість
    enhanced_image = tf.squeeze(enhanced_image) # Прибираємо batch

    print(f"Вхідний розмір: {original_image.shape}") # Вивід розмірів попереднього зображення
    print(f"Вихідний розмір: {enhanced_image.shape}") # Вивід розмірів отриманого зображення

    # ── Рисунок 8: Виведення та збереження результату ───────────────────────
    plt.rcParams['figure.figsize'] = [15, 10]
    fig, axes = plt.subplots(1, 2) # Одна строка, 2 картинки
    fig.tight_layout()

    # Виводимо оригінальне зображення
    plt.subplot(1, 2, 1)
    plot_image(tf.squeeze(original_image), title="Original") # Ліворуч

    # Виводимо покращене зображення
    plt.subplot(1, 2, 2)
    fig.tight_layout()
    plot_image(tf.squeeze(enhanced_image), "Super Resolution") # Праворуч

    # Зберігаємо результат
    if not os.path.exists(directory_path + "enhanced"):
        os.makedirs(directory_path + "enhanced")

    plt.savefig(directory_path + "enhanced/enhanced_" + image_name, bbox_inches="tight")
    plt.show()
    print(f"Збережено: {directory_path}enhanced/enhanced_{image_name}")

# ── Виклик функції process_image ──────────────────────────────────────────
# Для завдання лабораторної роботи - обробляємо 2-3 зображення:
process_image(LAB_PATH, "baboon.png")

# Покращення зображення
process_image(LAB_PATH, "lake_house.png")

# Покращення зображення
process_image(LAB_PATH, "anime_person.png")
