import os
import cv2
import numpy as np

# Вказуємо шлях до зображень
absolute_path = os.getcwd()
image_path = absolute_path + '/lab03/lane_markings.jpg'
video_path = absolute_path + '/lab03/854669-hd_1920_1080_30fps.mp4'

def wait_and_clear(task_num : float, next_task : str = "продовжити"):
    print(f"\n--- Натисніть ENTER (в консолі або на вікні зображення) щоб закрити завдання {task_num} і {next_task}) ---")
    cv2.waitKey(1)
    input()
    cv2.destroyAllWindows()

def draw_lines(img, lines, color=[255, 0, 0], thickness=7):
    """Calculates average slopes and draws unified left and right lane lines."""
    x_bottom_pos, x_upper_pos = [], []
    x_bottom_neg, x_upper_neg = [], []

    # y_bottom = 540 # Bottom of the image (adjust based on your image size)
    # y_upper = 315  # Top of the ROI (adjust based on your image size)
    y_bottom = img.shape[0]
    y_upper = int(img.shape[0] * 0.5)

    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                # Prevent division by zero
                if x2 - x1 == 0: 
                    continue
                
                slope = (y2 - y1) / (x2 - x1)
                
                # Test and filter values based on slope to separate left/right lanes
                if 0.5 < slope < 0.8:
                    b = y1 - slope * x1
                    x_bottom_pos.append((y_bottom - b) / slope)
                    x_upper_pos.append((y_upper - b) / slope)
                elif -0.8 < slope < -0.5:
                    b = y1 - slope * x1
                    x_bottom_neg.append((y_bottom - b) / slope)
                    x_upper_neg.append((y_upper - b) / slope)

    # Draw Positive (Right) Line
    if len(x_bottom_pos) > 0 and len(x_upper_pos) > 0:
        cv2.line(img, 
                 (int(np.mean(x_bottom_pos)), int(y_bottom)), 
                 (int(np.mean(x_upper_pos)), int(y_upper)), 
                 color, thickness)

    # Draw Negative (Left) Line
    if len(x_bottom_neg) > 0 and len(x_upper_neg) > 0:
        cv2.line(img, 
                 (int(np.mean(x_bottom_neg)), int(y_bottom)), 
                 (int(np.mean(x_upper_neg)), int(y_upper)), 
                 color, thickness)

def process_image(img_path):
    """Processes a single image step-by-step for demonstration."""
    print("Starting Step-by-Step Image Processing...")
    
    # 1. Load original image
    img = cv2.imread(img_path)
    if img is None:
        print(f"Error: Could not load image at {img_path}")
        return
    cv2.imshow('1 - Input image', img)
    wait_and_clear(1)

    # 2. Convert to Grayscale
    grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imshow('2 - Grayscale', grayscale)
    wait_and_clear(2)

    # 3. Apply Gaussian Blur
    blur_ksize = 9
    blur = cv2.GaussianBlur(grayscale, (blur_ksize, blur_ksize), 0)
    cv2.imshow('3 - Blurred', blur)
    wait_and_clear(3)

    # 4. Canny Edge Detection
    low_t = 50
    high_t = 150
    edges = cv2.Canny(blur, low_t, high_t)
    cv2.imshow('4 - Canny Edges', edges)
    wait_and_clear(4)

    # 4.1 Mask out car hood and the dust at the bottom
    h, w = edges.shape[:2]
    # Area: from 0% width and 83% height to the bottom-right corner
    cv2.rectangle(edges, (0, int(h * 0.83)), (w, h), 0, -1)
    cv2.imshow('4.1 - Edges without Hood', edges)
    wait_and_clear(4.1)

    # 4.2 Dilate edges to make lines thicker
    dilate_ksize = 20
    kernel_dilate = np.ones((dilate_ksize, dilate_ksize), np.uint8)
    dilated_edges = cv2.dilate(edges, kernel_dilate, iterations=1)
    cv2.imshow('4.2 - Dilated Edges', dilated_edges)
    wait_and_clear(4.2)

    # 5. Region of Interest (ROI) Mask
    # h, w already obtained from edges.shape above
    # Using a hexagon ROI to capture the full width at the bottom and sides
    # and taper toward the horizon to focus on the lanes.
    vertices = np.array([
        [
            (int(w * 0.3), h),               # bottom-left
            (w, h),                          # bottom-right
            (w, int(h * 0.7)),               # mid-right (30% up from bottom)
            (int(w * 0.66), int(h * 0.42)),  # top-right
            (int(w * 0.62), int(h * 0.42)),  # top-left
            (int(w * 0.3), int(h * 0.8))     # mid-left (80% up from bottom)
        ]
    ], dtype=np.int32)

    mask = np.zeros_like(edges)
    ignore_mask_color = 255
    masked_poly = cv2.fillPoly(mask, vertices, ignore_mask_color)
    cv2.imshow('5 - Masked Poly', masked_poly)
    wait_and_clear(5.1)

    masked_edges = cv2.bitwise_and(dilated_edges, mask) # Apply mask to dilated edges
    cv2.imshow('5 - Masked Edges', masked_edges)
    wait_and_clear(5.2)

    # 6. Hough Transform
    rho = 3
    theta = np.pi / 180
    threshold = 15
    min_line_len = 150
    max_line_gap = 60
    
    lines = cv2.HoughLinesP(masked_edges, rho, theta, threshold, np.array([]), 
                            minLineLength=min_line_len, maxLineGap=max_line_gap)
    
    # 7. Draw Lines on the original image
    result_img = img.copy()
    draw_lines(result_img, lines)
    cv2.imshow('6 - Final Result with Hough Lines', result_img)
    wait_and_clear(6)

def process_video(video_path):
    """Applies the pipeline to a video stream."""
    print("Starting Video Processing... Press 'q' to stop.")
    video_capture = cv2.VideoCapture(video_path)
    
    while video_capture.isOpened():
        ret, frame = video_capture.read()
        if ret:
            # Inline processing for the video frame
            blur_ksize = 9
            grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(grayscale, (blur_ksize, blur_ksize), 0)
            edges = cv2.Canny(blur, 50, 150)
            
            h, w = edges.shape[:2]
            # Mask out car hood to prevent it from interfering with Hough Transform
            cv2.rectangle(edges, (0, int(h * 0.83)), (w, h), 0, -1)

            # Dilate edges to make lines thicker for better Hough Transform detection
            dilate_ksize = 20
            kernel_dilate = np.ones((dilate_ksize, dilate_ksize), np.uint8)
            dilated_edges = cv2.dilate(edges, kernel_dilate, iterations=1)

            vertices = np.array([
                [
                    (int(w * 0.3), h),               # bottom-left
                    (w, h),                          # bottom-right
                    (w, int(h * 0.7)),               # mid-right (30% up from bottom)
                    (int(w * 0.66), int(h * 0.42)),  # top-right
                    (int(w * 0.62), int(h * 0.42)),  # top-left
                    (int(w * 0.3), int(h * 0.8))     # mid-left (80% up from bottom)
                ]
            ], dtype=np.int32)

            mask = np.zeros_like(edges)
            cv2.fillPoly(mask, vertices, 255) # Fill the mask with white
            masked_edges = cv2.bitwise_and(dilated_edges, mask) # Apply mask to dilated edges
            
            lines = cv2.HoughLinesP(masked_edges, 3, np.pi / 180, 15, np.array([]), 
                                    minLineLength=150, maxLineGap=60)
            
            draw_lines(frame, lines)
            
            cv2.imshow('Video Frame', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break
            
    video_capture.release()
    cv2.destroyAllWindows()

# Виконання
process_image(image_path)
process_video(video_path)
