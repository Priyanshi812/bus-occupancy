import cv2
import pandas as pd
from ultralytics import YOLO
from tracker import Tracker
import cvzone
import numpy as np
import streamlit as st
from PIL import Image

# Streamlit UI setup
st.set_page_config(layout="wide")
st.title("People Counting System")

# Initialize session state
if 'going_out_count' not in st.session_state:
    st.session_state.going_out_count = 0
if 'going_in_count' not in st.session_state:
    st.session_state.going_in_count = 0

# Load model and class labels
model = YOLO('yolov8s.pt')
with open("coco.txt", "r") as my_file:
    class_list = my_file.read().split("\n")

# Define detection zones
entry_zone = [(494, 289), (505, 499), (578, 496), (530, 292)]
exit_zone = [(548, 290), (600, 496), (637, 493), (574, 288)]

# Initialize tracker and counters
tracker = Tracker()
going_out, going_in = {}, {}
counter_out_ids, counter_in_ids = [], []

# Streamlit layout
col1, col2 = st.columns([3, 1])
video_placeholder = col1.empty()

# Counter display
with col2:
    st.subheader("Counters")
    out_display = st.empty()
    in_display = st.empty()
    st.markdown("---")
    

# Read video
cap = cv2.VideoCapture("p.mp4")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        st.warning("Video processing completed!")
        break

    frame = cv2.resize(frame, (1020, 500))

    # YOLOv8 Detection
    results = model.predict(frame, verbose=False)
    boxes = results[0].boxes.data
    px = pd.DataFrame(boxes).astype("float")

    detections = []
    for _, row in px.iterrows():
        x1, y1, x2, y2, _, class_id = map(int, row)
        if class_list[class_id] == "person":
            detections.append([x1, y1, x2, y2])

    # Track people
    tracked_ids = tracker.update(detections)

    for x1, y1, x2, y2, id in tracked_ids:
        center = (x2, y2)

        if cv2.pointPolygonTest(np.array(exit_zone, np.int32), center, False) >= 0:
            going_out[id] = center

        if id in going_out:
            if cv2.pointPolygonTest(np.array(entry_zone, np.int32), center, False) >= 0:
                if id not in counter_out_ids:
                    counter_out_ids.append(id)
                    st.session_state.going_out_count = len(counter_out_ids)

        if cv2.pointPolygonTest(np.array(entry_zone, np.int32), center, False) >= 0:
            going_in[id] = center

        if id in going_in:
            if cv2.pointPolygonTest(np.array(exit_zone, np.int32), center, False) >= 0:
                if id not in counter_in_ids:
                    counter_in_ids.append(id)
                    st.session_state.going_in_count = len(counter_in_ids)

        # Draw detections
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
        cvzone.putTextRect(frame, str(id), (x1, y1), 1, 1)

    # Draw on frame
    #cvzone.putTextRect(frame, f'OUT: {len(counter_out_ids)}', (50, 60), 2, 2)
    #cvzone.putTextRect(frame, f'IN: {len(counter_in_ids)}', (50, 120), 2, 2)
    cv2.polylines(frame, [np.array(entry_zone, np.int32)], True, (0, 255, 0), 2)
    cv2.polylines(frame, [np.array(exit_zone, np.int32)], True, (0, 0, 255), 2)

    # Update video
    video_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)
    # Update metrics
    out_display.metric("Out", st.session_state.going_out_count)
    in_display.metric("In", st.session_state.going_in_count)

cap.release()
