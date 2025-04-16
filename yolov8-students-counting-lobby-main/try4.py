import cv2
import pandas as pd
from ultralytics import YOLO
from tracker import *
import cvzone
import numpy as np

model = YOLO('yolov8s.pt')

def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:  
        point = [x, y]
        print(point)

cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)

cap = cv2.VideoCapture('new2.mp4')

my_file = open("coco.txt", "r")
data = my_file.read()
class_list = data.split("\n")

tracker = Tracker()
area1 = [(494, 289), (505, 499), (578, 496), (530, 292)]

counted = set()  # Set to store IDs already counted

while True:    
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (1020, 500))
    results = model.predict(frame)
    a = results[0].boxes.data
    px = pd.DataFrame(a).astype("float")

    person_list = []
    for index, row in px.iterrows():
        x1, y1, x2, y2, _, d = row
        x1, y1, x2, y2, d = int(x1), int(y1), int(x2), int(y2), int(d)
        class_name = class_list[d]
        if class_name == 'person':
            person_list.append([x1, y1, x2, y2])

    bbox_idx = tracker.update(person_list)
    for bbox in bbox_idx:
        x3, y3, x4, y4, id = bbox
        # Bottom-center point
        cx = (x3 + x4) // 2
        cy = y4
        result2 = cv2.pointPolygonTest(np.array(area1, np.int32), (cx, cy), False)

        if result2 >= 0 and id not in counted:
            counted.add(id)
            print(f"Person {id} counted")

        if id in counted:
            cv2.circle(frame, (cx, cy), 7, (255, 0, 255), -1)
            cv2.rectangle(frame, (x3, y3), (x4, y4), (255, 0, 0), 2)
            cvzone.putTextRect(frame, f'{id}', (x3, y3), 1, 1)

    in_c = len(counted)
    cvzone.putTextRect(frame, f'IN_C: {in_c}', (50, 60), 2, 2)

    cv2.polylines(frame, [np.array(area1, np.int32)], True, (0, 255, 0), 2)
    cv2.imshow("RGB", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
