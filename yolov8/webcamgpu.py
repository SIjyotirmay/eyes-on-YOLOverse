from ultralytics import YOLO
import cv2
import cvzone
import math
import torch



# Check if GPU is available
print("CUDA Available:", torch.cuda.is_available())

# Open webcam
cap = cv2.VideoCapture(0)
# Optional: use a video file instead
# cap = cv2.VideoCapture("./Videos/video.mp4")

# Set webcam resolution
cap.set(3, 1000)
cap.set(4, 720)

# Load YOLOv8 model and move to GPU
model = YOLO('yolov8n.pt').to('cuda')
# Optional: use half precision for speed boost
model.model.half()

# COCO class names
classNames = [
    "person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
    "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
    "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
    "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
    "teddy bear", "hair drier", "toothbrush"
]

while True:
    success, img = cap.read()
    if not success:
        break

    # Convert image to float16 for speed (if using half precision)
    img_gpu = torch.from_numpy(img).to('cuda').half()

    # Run YOLOv8 on the frame
    results = model(img, stream=True)

    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Get box coordinates
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1

            # Draw corner rectangle
            cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=2, colorC=(255, 0, 255))

            # Confidence
            conf = math.ceil((box.conf[0] * 100)) / 100

            # Class name
            cls = int(box.cls[0])
            class_name = classNames[cls] if cls < len(classNames) else "Unknown"

            # Put label
            cvzone.putTextRect(img, f'{class_name} {conf}', (max(0, x1), max(30, y1)), scale=1, thickness=1)

    # Show the frame
    cv2.imshow("YOLOv8 Detection (GPU)", img)

    # Break on ESC key
    if cv2.waitKey(1) == 27:
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
