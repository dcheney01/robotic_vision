from ultralytics import YOLO
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

model = YOLO("yolov8s.pt")

results = model.train(data='datasets/data.yaml', epochs=1, verbose=False)