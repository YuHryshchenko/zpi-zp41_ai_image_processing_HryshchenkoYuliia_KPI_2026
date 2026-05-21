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
# ────────── Частина 2 – Створення зображення на основі іншого зображення ──────────
# ────────────────  (Image-to-Image)   ─────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────────────
print("Почали Частину 2 – Створення зображення на основі іншого зображення")

if torch.backends.mps.is_available():
    torch.mps.empty_cache()

# sd-turbo замість SD v1.5: ~10x швидше завдяки дистиляції
pipeline = AutoPipelineForImage2Image.from_pretrained(
#    "stabilityai/sd-turbo",
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float32,   # float32 стабільніше на MPS (M-серія)
    use_safetensors=True,
).to(device)

pipeline.enable_attention_slicing()

# Важливо: Переводимо в RGB та змінюємо розмір на кратний 8. 
# Для Stable Diffusion ідеально підходить 512x512.
url = LAB_PATH + "squirrel.jpg"
init_image = load_image(url).convert("RGB").resize((512, 512))

prompt = "squirrel wizard, lord of the rings, detailed, fantasy, cute, adorable, Pixar, Disney"

# sd-turbo: strength * num_inference_steps має бути >= 1
# тому при strength=0.5 потрібно >= 2 кроки; 8-10 кроків — оптимум
image = pipeline(
    prompt=prompt,
    image=init_image,
    strength=0.9,            # нижче ніж 0.8 — краще зберігає структуру
    num_inference_steps=50,  # sd-turbo достатньо 8-10 кроків (stable-diffusion-v1-5 краще працює за 50)
#    guidance_scale=0.0,     # sd-turbo не потребує classifier-free guidance
).images[0]

grid_image = make_image_grid(images=[init_image, image], rows=1, cols=2)
grid_image.save(LAB_PATH + "output/squirrel_wizard.png")