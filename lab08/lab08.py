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

# ──────────────────────────────────────────────────────────────────────────────────
# ────────── Частина 4 – Створення зображення на основі іншого зображення ──────────
# ────────────────  (Image-to-Image)   ─────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────────────
print("Почали Частину 4 – Створення зображення на основі іншого зображення")
pipeline = AutoPipelineForImage2Image.from_pretrained(
    model_id,
#    torch_dtype=torch.float16,
    torch_dtype=torch.float32,
#    use_safetensors=True,
    requires_safety_checker=False,
    safety_checker=None
).to(device) # explicitly move to mps

# Removed cpu_offload() and replaced with attention slicing for Mac
# pipeline.enable_attention_slicing()

url = LAB_PATH + "squirrel.jpg"
init_image = load_image(url)

prompt = "squirrel wizard, lord of the rings, detailed, fantasy, cute, adorable, Pixar, Disney"

image = pipeline(prompt, image=init_image).images[0]
grid_image = make_image_grid(images=[init_image, image], rows=1, cols=2)

grid_image.save(LAB_PATH + "output/squirrel_wizard.png")


# ──────────────────────────────────────────────────────────────────────────────────
# ────────── Частина 4 – Створення зображення на основі іншого зображення ──────────
# ────────────────  (Image-to-Image)   ─────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────────────
# device = "cpu"
print("Почали Частину 4 – Створення зображення на основі іншого зображення")

# 1. Очищаємо кеш пам'яті перед завантаженням нової моделі (Критично для Mac)
if torch.backends.mps.is_available():
    torch.mps.empty_cache()

pipeline = AutoPipelineForImage2Image.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16,
#    torch_dtype=torch.float32,
    use_safetensors=True,
    requires_safety_checker=False,
    safety_checker=None
).to(device)

pipeline.enable_attention_slicing()

url = LAB_PATH + "squirrel.jpg"
init_image = load_image(url)

# 2. КРИТИЧНО ВАЖЛИВО: Переводимо в RGB та змінюємо розмір на кратний 8. 
# Для Stable Diffusion v1.5 ідеально підходить 512x512.
init_image = init_image.convert("RGB").resize((256, 256))

prompt = "squirrel wizard fantasy cute"

# 3. Додаємо параметри strength та num_inference_steps для кращого контролю
image = pipeline(
    prompt=prompt, 
    image=init_image,
    strength=0.8, # Наскільки сильно змінювати оригінал (від 0.0 до 1.0)
    num_inference_steps=30
).images[0]

grid_image = make_image_grid(images=[init_image, image], rows=1, cols=2)
grid_image.save(LAB_PATH + "output/squirrel_wizard.png")