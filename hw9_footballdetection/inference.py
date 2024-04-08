from ultralytics import YOLO
import cv2 

model = YOLO("hw9_footballdetection/weights/best (1).pt")

video_path = "hw9_footballdetection/datasets/Test 2.mp4"

cap = cv2.VideoCapture(video_path)
writer = cv2.VideoWriter("hw9_footballdetection/Test2_output.mp4", cv2.VideoWriter_fourcc(*'mp4v'), 30, (640, 480))

colors = [(0, 0, 255), (0, 255, 0)]
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.resize(frame, (640, 480))

    # cv2.imshow("frame", frame)
    results = model.predict(frame, show=False)

    for result in results[0].boxes:
        x1 = int(result.xywh[0][0].item())
        y1 = int(result.xywh[0][1].item())
        w = int(result.xywh[0][2].item())
        h = int(result.xywh[0][3].item())

        if result.cls[0] == 0:
            cv2.rectangle(frame, (x1, y1), (x1+w, y1+h), colors[0], 2)
        elif result.cls[0] == 1:
            cv2.rectangle(frame, (x1, y1), (x1+w, y1+h), colors[1], 2)

    writer.write(frame)
    # cv2.imshow("frame", frame)

    # if cv2.waitKey(0) & 0xFF == ord('q'):
    #     break
    

cap.release()
writer.release()
cv2.destroyAllWindows()

