from ultralytics import YOLO

model = YOLO('yolov8m-cls.pt')

results = model.train(data='/home/daniel/software/robotic_vision/hw3/data/Oyster Shell', epochs=250, verbose=False, imgsz=640, device=[0,1], patience=0)

# results = model.train(data='/home/daniel/software/robotic_vision/hw3/data/8 Fish Species', epochs=50, verbose=False, imgsz=640, device=[0,1])