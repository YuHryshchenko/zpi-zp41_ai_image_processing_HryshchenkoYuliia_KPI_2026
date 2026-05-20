import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers
import matplotlib.pyplot as plt
import os
import time

ABSOLUTE_PATH = os.getcwd()
LAB_PATH = ABSOLUTE_PATH + "/lab06/"

# 1. Завантаження датасету
(train_images, train_labels), (test_images, test_labels) = keras.datasets.cifar10.load_data()

# 2. Ініціалізація назв класів
CLASS_NAMES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# 3. Отримання даних для валідації (останні 500 зображень з навчального набору)
validation_images, validation_labels = train_images[:500], train_labels[:500]
train_images, train_labels = train_images[500:], train_labels[500:]

# 4. Перетворення на tf.data.Dataset
train_ds = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
test_ds = tf.data.Dataset.from_tensor_slices((test_images, test_labels))
validation_ds = tf.data.Dataset.from_tensor_slices((validation_images, validation_labels))

# 5. Функція попередньої обробки
def process_images(image, label):
    # Зміна розміру до 227x227 (як очікує оригінальний AlexNet)
    image = tf.image.resize(image, (227, 227))
    # Нормалізація (приведення до діапазону [0, 1])
    image = tf.cast(image, tf.float32) / 255.0
    return image, label

# 6. Створення пайплайну (обробка, перемішування, пакетування)
train_ds = train_ds.map(process_images).shuffle(buffer_size=1000).batch(32)
validation_ds = validation_ds.map(process_images).batch(32)
test_ds = test_ds.map(process_images).batch(32)

# 7. Архітектура AlexNet
model = models.Sequential([
    layers.Conv2D(96, (11, 11), strides=4, activation='relu', input_shape=(227, 227, 3)),
    layers.MaxPooling2D((3, 3), strides=2),
    layers.Conv2D(256, (5, 5), padding='same', activation='relu'),
    layers.MaxPooling2D((3, 3), strides=2),
    layers.Conv2D(384, (3, 3), padding='same', activation='relu'),
    layers.Conv2D(384, (3, 3), padding='same', activation='relu'),
    layers.Conv2D(256, (3, 3), padding='same', activation='relu'),
    layers.MaxPooling2D((3, 3), strides=2),
    layers.Flatten(),
    layers.Dense(4096, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(4096, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(10, activation='softmax')
])

# 8. Налаштування TensorBoard
root_logdir = LAB_PATH + "/logs"
def get_run_logdir():
    import time
    run_id = time.strftime("run_%Y_%m_%d-%H_%M_%S")
    return os.path.join(root_logdir, run_id)

run_logdir = get_run_logdir()
tensorboard_cb = keras.callbacks.TensorBoard(run_logdir)

# 9. Компіляція мережі
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Опис мережі
model.summary()

# 10. Навчання
history = model.fit(train_ds, epochs=2, validation_data=validation_ds, callbacks=[tensorboard_cb])

# 11. Оцінка
test_loss, test_acc = model.evaluate(test_ds)
print(f"Точність на тестових даних: {test_acc}")
