from flask import Flask, render_template
from flask_socketio import SocketIO
import cv2
from ultralytics import YOLO
from tracker import *
import numpy as np
import threading

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins='*')

# Load YOLO model
model = YOLO('yolov8s.pt')

# COCO class list
class_list = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
    'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
    'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
    'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
    'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
    'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
    'toothbrush'
]

# Tracker and area
tracker = Tracker()
counting_area = [(440, 140), (520, 140), (520, 480), (440, 480)]

# Load video
cap = cv2.VideoCapture('/Users/dhruvibhalodiya/Desktop/bus/bus-passengers-counter-main/new2.mp4')

@app.route('/')
def home():
    return render_template("tracker.html")

def process_video():
    while True:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        frame = cv2.resize(frame, (1020, 500))
        results = model.predict(frame, conf=0.5)
        detections = results[0].boxes.data.cpu().numpy()

        person_list = []
        for row in detections:
            x1, y1, x2, y2, _, class_id = row
            if class_list[int(class_id)] == 'person':
                person_list.append([int(x1), int(y1), int(x2), int(y2)])

        tracked = tracker.update(person_list)

        current_ids_in_area = set()
        for track_id, box in tracked.items():
            x1, y1, x2, y2 = box
            cx = x1 + (x2 - x1) // 2
            cy = y2
            if cv2.pointPolygonTest(np.array(counting_area, np.int32), (cx, cy), False) >= 0:
                current_ids_in_area.add(track_id)

        occupied = len(current_ids_in_area)
        total = 40
        available = total - occupied
        capacity = int((occupied / total) * 100)

        # Send data to frontend
        socketio.emit('update_data', {
            'occupied': occupied,
            'available': available,
            'capacity': capacity
        })

        cv2.waitKey(1)

@socketio.on('connect')
def handle_connect():
    print("Client connected")

if __name__ == '__main__':
    threading.Thread(target=process_video).start()
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)
