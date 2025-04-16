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

class_list = model.model.names
tracker = Tracker()

area1 = [(494, 289), (505, 499), (578, 496), (530, 292)]  # IN area
area2 = [(548, 290), (600, 496), (637, 493), (574, 288)]  # OUT area

counter_in = []
counter_out = []
status = {}  # Keeps track of each ID's last known zone: 'in', 'out', or None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (1020, 500))
    results = model.predict(frame, conf=0.25)
    a = results[0].boxes.data
    px = pd.DataFrame(a).astype("float")

    person_list = []
    for index, row in px.iterrows():
        x1, y1, x2, y2, _, cls_id = row
        x1, y1, x2, y2, cls_id = int(x1), int(y1), int(x2), int(y2), int(cls_id)
        class_name = class_list[cls_id]
        if class_name == 'person':
            person_list.append([x1, y1, x2, y2])

    bbox_idx = tracker.update(person_list)

    for bbox in bbox_idx:
        x3, y3, x4, y4, id = bbox

        # Check zone
        point = (x4, y4)
        in_zone = cv2.pointPolygonTest(np.array(area1, np.int32), point, False) >= 0
        out_zone = cv2.pointPolygonTest(np.array(area2, np.int32), point, False) >= 0

        prev_zone = status.get(id)

        if prev_zone is None:
            # Set initial status
            if in_zone:
                status[id] = 'in'
            elif out_zone:
                status[id] = 'out'
        else:
            if prev_zone == 'out' and in_zone:
                # OUT -> IN means entry
                if id not in counter_in:
                    counter_in.append(id)
                    print(f'Person {id} entered')
                status[id] = 'in'

            elif prev_zone == 'in' and out_zone:
                # IN -> OUT means exit
                if id not in counter_out:
                    counter_out.append(id)
                    print(f'Person {id} exited')
                status[id] = 'out'

        # Draw everything
        cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 255, 0), 2)
        cvzone.putTextRect(frame, f'{id}', (x3, y3), 1, 1)
        cv2.circle(frame, point, 5, (255, 0, 255), -1)

    # Display counts
    cvzone.putTextRect(frame, f'IN Count: {len(counter_in)}', (50, 60), 2, 2)
    cvzone.putTextRect(frame, f'OUT Count: {len(counter_out)}', (50, 120), 2, 2)

    # Draw the zones
    cv2.polylines(frame, [np.array(area1, np.int32)], True, (0, 255, 0), 2)
    cv2.polylines(frame, [np.array(area2, np.int32)], True, (0, 0, 255), 2)

    cv2.imshow("RGB", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
