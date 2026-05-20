import os
from PIL import Image

ABSOLUTE_PATH = os.getcwd()
LAB_PATH = ABSOLUTE_PATH + "/lab07/"

# Крок 1: Візьмемо будь-яке чітке зображення (наприклад, 512×512)
# Крок 2: Зменшимо його методом bicubic у 4 рази - це і є "правильний" вхід для ESRGAN
for orig_image in [
#        "baboon01.png",
        "lake_house01.png",
#        "anime_person01.png"
    ]:
    high_res = Image.open(LAB_PATH + orig_image).convert("RGB")
    w, h = high_res.size
    low_res = high_res.resize((w // 8, h // 8), Image.BICUBIC)
    low_res.save(LAB_PATH + orig_image + "_conv001.png")

# Тепер запускаємо process_image("input_image.png")
# і порівнюємо результат з оригіналом high_res
