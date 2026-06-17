from ultralytics import YOLO
import cv2

import torch
print(torch.cuda.is_available())  # should print True
print(torch.cuda.get_device_name(0))


model = YOLO('../yolo-weights/yolov8n.pt')
model.to('cuda')
results = model("videos/bikes.mp4",show = True,stream=True,device='cuda')

for r in results:
    print(r.boxes)

cv2.destroyAllWindows()

