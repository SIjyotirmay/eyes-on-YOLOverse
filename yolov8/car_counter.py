from ultralytics import YOLO
import cv2
import cvzone
import math


#cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture("./videos/cars.mp4")

cap.set(3,1000)
cap.set(4,720)
model = YOLO('yolov8n.pt').to('cuda')

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
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

mask = cv2.imread('./images/mask.png')

while True:
    sucess, img = cap.read()
    mask = cv2.resize(mask, (img.shape[1], img.shape[0]))  # width, height
    imgRegion = cv2.bitwise_and(img,mask)
    results = model(imgRegion, stream=True)
    for r in results:
        boxes  = r.boxes
        for box in boxes:

            #bounding box
            x1,y1,x2,y2=box.xyxy[0]
            x1,y1,x2,y2 = int(x1),int(y1),int(x2),int(y2)
            #print(x1,y1,x2,y2)
            #cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,255),thickness=3)
            w ,h = x2-x1, y2-y1
            #bbox = int(x1),int(y1),int(w),int(h)
            cvzone.cornerRect(img,(x1,y1,w,h),l=9)

            #confidence
            conf = math.ceil((box.conf[0]*100)) /100
            #print(conf)
            #cvzone.putTextRect(img,f'{conf}',(max(0,x1),max(30,y1)))
            
            #class name
            cls  = int(box.cls[0])
            currentclass = classNames[cls]

            if currentclass == "car" or currentclass == "bus" or currentclass == "motorbike"  :
                cvzone.putTextRect(img, f'{classNames[cls]} {conf}',(max(0,x1),max(30,y1)),scale=0.6,thickness=1,offset=3 )



    cv2.imshow("image",img)
    #
    # cv2.imshow("imageRegion",imgRegion)
    cv2.waitKey(1)