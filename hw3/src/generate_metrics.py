import ultralytics

model = ultralytics.YOLO('/home/daniel/software/robotic_vision/runs/classify/train22/weights/best.pt')

# model = ultralytics.YOLO('/home/daniel/software/robotic_vision/hw3/runs/classify/fish1/weights/best.pt')

model.val()

