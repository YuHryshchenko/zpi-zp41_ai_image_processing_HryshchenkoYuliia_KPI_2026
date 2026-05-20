import os
from imageai.Detection import VideoObjectDetection

USE_LEGACY_H5 = False
ABSOLUTE_PATH = os.getcwd()
LAB_PATH = ABSOLUTE_PATH + "/lab05/"

H5_MODEL_PATH = LAB_PATH + "/yolo-tiny.h5"
PT_MODEL_PATH = LAB_PATH + "/tiny-yolov3.pt"
INPUT_VIDEO_PATH = LAB_PATH + "/21438-317457608_medium.mp4"
OUTPUT_VIDEO_PATH = LAB_PATH + "/output/new_video" # ImageAI автоматично додасть розширення (наприклад, .avi)

def run_video_detection():
    print("[VIDEO] Початок обробки відео (це може зайняти час)...")
    
    if USE_LEGACY_H5:
        model_path = H5_MODEL_PATH
    else:
        model_path = PT_MODEL_PATH

    # Ініціалізація відеотетектора
    video_detector = VideoObjectDetection()
    
    # Щоб використовувати власноруч навчену модель (Custom),
    # відповідно до документації ImageAI v3, розкоментуйте наступні 3 рядки 
    # та закоментуйте стандартний ObjectDetection() вище:
    # from imageai.Detection.Custom import CustomVideoObjectDetection
    # video_detector = CustomVideoObjectDetection()
    # video_detector.setJsonPath("./models/detection_config.json")

    video_detector.setModelTypeAsTinyYOLOv3()
    video_detector.setModelPath(model_path)
    video_detector.loadModel()

    # Обробка відеокадрів
    video_detector.detectObjectsFromVideo(
        input_file_path=INPUT_VIDEO_PATH,
        output_file_path=OUTPUT_VIDEO_PATH,
        frames_per_second=20,
        log_progress=True
    )
    
    print("[VIDEO] Відео успішно оброблено та збережено!")