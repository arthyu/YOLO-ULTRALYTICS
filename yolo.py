from ultralytics import YOLO
import cv2

model = YOLO('../yolo-weights/yolov8n.pt')
results = model("images/img2.jpg",show = True)

img = results[0].plot() 
cv2.imshow("YOLOv8 Detection", img)

cv2.waitKey(0)
