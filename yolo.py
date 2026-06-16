from ultralytics import YOLO
import cv2

model = YOLO('../yolo-weights/yolov8n.pt')
results = model("videos/bikes.mp4",show = True,stream=True)

for r in results:
    pass

cv2.destroyAllWindows()
