import cv2
import pandas as pd
from ultralytics import YOLO
from tracker import *
import cvzone
import numpy as np

# Load YOLOv8 model
model = YOLO('yolov8s.pt')

# Load COCO class names
with open("coco.txt", "r") as f:
    class_list = f.read().split("\n")

# Mouse pointer tool to check coordinates
def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        print(f"Mouse at: {(x, y)}")

cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)

# Load video
cap = cv2.VideoCapture('new2.mp4')

# Entry/Exit zones

# These coordinates are based on the visible entrance in your image
area1 = [(1000, 270), (1010, 490), (900, 490), (890, 270)]  # INSIDE
area2 = [(450, 300), (460, 480), (700, 480), (710, 300)]    # OUTSIDE


# Trackers and counters
tracker = Tracker()
going_in = {}
going_out = {}
counter_in = []
counter_out = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (1020, 500))
    results = model.predict(frame)
    detections = pd.DataFrame(results[0].boxes.data).astype("float")

    person_list = []

    # Get person detections only
    for _, row in detections.iterrows():
        x1, y1, x2, y2 = int(row[0]), int(row[1]), int(row[2]), int(row[3])
        class_id = int(row[5])
        class_name = class_list[class_id]

        if class_name == 'person':
            person_list.append([x1, y1, x2, y2])

    # Update tracker with current detections
    tracked_objects = tracker.update(person_list)

    for x1, y1, x2, y2, obj_id in tracked_objects:
        cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)  # Center point of bbox

        # Check if entering from area1 to area2 (going out)
        if cv2.pointPolygonTest(np.array(area1, np.int32), (cx, cy), False) >= 0:
            going_out[obj_id] = (cx, cy)
        if obj_id in going_out:
            if cv2.pointPolygonTest(np.array(area2, np.int32), (cx, cy), False) >= 0:
                if obj_id not in counter_out:
                    counter_out.append(obj_id)
                    print(f"Person {obj_id} went OUT")

        # Check if entering from area2 to area1 (going in)
        if cv2.pointPolygonTest(np.array(area2, np.int32), (cx, cy), False) >= 0:
            going_in[obj_id] = (cx, cy)
        if obj_id in going_in:
            if cv2.pointPolygonTest(np.array(area1, np.int32), (cx, cy), False) >= 0:
                if obj_id not in counter_in:
                    counter_in.append(obj_id)
                    print(f"Person {obj_id} came IN")

        # Drawing bbox and ID
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)
        cvzone.putTextRect(frame, f'ID: {obj_id}', (x1, y1), 1, 1)
        cv2.circle(frame, (cx, cy), 5, (255, 0, 255), -1)

    # Show counters
    cvzone.putTextRect(frame, f'OUT_C: {len(counter_out)}', (50, 60), 2, 2)
    cvzone.putTextRect(frame, f'IN_C: {len(counter_in)}', (50, 120), 2, 2)

    # Draw zones
    cv2.polylines(frame, [np.array(area1, np.int32)], True, (0, 255, 0), 2)
    cv2.polylines(frame, [np.array(area2, np.int32)], True, (0, 255, 0), 2)

    cv2.imshow("RGB", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
