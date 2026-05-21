import os
import torch
from diffusers import StableDiffusionPipeline
from diffusers import AutoPipelineForImage2Image
from diffusers.utils import make_image_grid, load_image
from diffusers.pipelines.stable_diffusion import safety_checker

# ──────────────────────────────────────────────────────────────────────────────────
# Лабораторна робота №8 - Генерування зображень за допомогою Stable Diffusion v1-5
# ──────────────────────────────────────────────────────────────────────────────────

ABSOLUTE_PATH = os.getcwd()
LAB_PATH = ABSOLUTE_PATH + "/lab08/"

# Determine the device (fallback to cpu if mps is somehow unavailable)
device = "mps" if torch.backends.mps.is_available() else "cpu"
# ──────────────────────────────────────────────────────────────────────────────────
# ────────────── Частина 1 – Імпорт моделі та створення конвеєра ───────────────────
# ──────────────────────────────────────────────────────────────────────────────────
print("Почали Частину 1 – Імпорт моделі та створення конвеєра")
model_id = "runwayml/stable-diffusion-v1-5"

def sc(self, clip_input, images) : return images, [False for i in images]

# edit the StableDiffusionSafetyChecker class so that, when called, it just returns the images and an array of True values
safety_checker.StableDiffusionSafetyChecker.forward = sc

# Changed from "cuda" to device ("mps")
pipe = StableDiffusionPipeline.from_pretrained(
    model_id, 
#    torch_dtype=torch.float16,
    torch_dtype=torch.float32,
    requires_safety_checker=False,
    safety_checker=None
).to(device)

# Recommended for Apple Silicon to manage unified memory efficiently
# pipe.enable_attention_slicing()
# pipe.safety_checker=None
pipe.safety_checker=lambda images, clip_input: (images, [False] * len(images))
pipe.requires_safety_checker=False

# ──────────────────────────────────────────────────────────────────────────────────
# ── Частина 2 – Створення зображення за промтом "Білка з горішком на дереві" ──────
# ──────────────────────────────────────────────────────────────────────────────────
print("Почали Частину 2 – Створення зображення за промтом 'Білка з горішком на дереві'")
#prompt = "a photo of a squirrel with a nut on tree"
prompt = "squirrel with a nut on tree"
image = pipe(prompt).images[0]

# Зберігаємо результат
if not os.path.exists(LAB_PATH + "output"):
    os.makedirs(LAB_PATH + "output")

image.save(LAB_PATH + "output/squirrel_with_nut.png")

# ──────────────────────────────────────────────────────────────────────────────────
# Частина 3 – Створення зображення за промтом "Кішко-дівчина" ──────────────────────
#  ─────────  з кількома параметрами:  ─────────────────────────────────────────────
#  ───────────  • height, width  — розмір у пікселях (кратний 8)  ──────────────────
#  ───────────  • guidance_scale — наскільки точно слідувати промту  ───────────────
#  ───────────  • negative_prompt — що модель НЕ повинна малювати  ─────────────────
# ──────────────────────────────────────────────────────────────────────────────────
print("Почали Частину 3 – Створення зображення за промтом 'Кішко-дівчина'")
#prompt = "cat-girl"
prompt = "cat girl from batman"
negative_prompt = "dog, man, strong"
image = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    guidance_scale=6,
    height=768,
    width=512
).images[0]

image.save(LAB_PATH + "output/cat_girl.png")
