import ultralytics

# model = ultralytics.YOLO('/home/daniel/software/robotic_vision/hw3/runs/classify/train/weights/best.pt')

model = ultralytics.YOLO('/home/daniel/software/robotic_vision/hw3/runs/classify/fish1/weights/best.pt')

model.val()

