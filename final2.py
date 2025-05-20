import os
import subprocess
import sys

try:
    import cv2
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "opencv-python-headless"])
    import cv2

import cv2
import pandas as pd
from ultralytics import YOLO
from tracker import Tracker_simp
import cvzone
import numpy as np
import streamlit as st
from PIL import Image
from tracker1 import Tracker_adv




# Set page layout
st.set_page_config(layout="wide")
st.sidebar.title("Navigation")

# Route selection
route = st.sidebar.radio("Select Route", ["Route 1", "Route 2", "Route 3"])

# Load model and class list
model = YOLO('yolov8s.pt')
with open("coco.txt", "r") as my_file:
    class_list = my_file.read().split("\n")

# Initialize session state counters
for key in ['route1_in', 'route1_out', 'route2_in', 'route2_out']:
    if key not in st.session_state:
        st.session_state[key] = 0

# Define zones (change per route if needed)
entry_zone = [(494, 289), (505, 499), (578, 496), (530, 292)]
exit_zone = [(548, 290), (600, 496), (637, 493), (574, 288)]

# Function for Routes 1 & 2 (with Streamlit session state tracking)
def run_people_counter(video_path, in_key, out_key):
    st.markdown(f"<h1 style='text-align:center;'> Bus Occupancy system - {route}</h1>", unsafe_allow_html=True)
    st.markdown("---")

    tracker = Tracker_simp()
    going_out, going_in = {}, {}
    counter_out_ids, counter_in_ids = [], []

    col1, col2 = st.columns([3, 1])
    video_placeholder = col1.empty()

    with col2:
        st.subheader("📊 Counters")
        out_display = st.empty()
        in_display = st.empty()
        st.markdown("---")

    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.success("✅ Video processing completed!")
            break

        frame = cv2.resize(frame, (1020, 500))
        results = model.predict(frame, verbose=False)
        boxes = results[0].boxes.data
        px = pd.DataFrame(boxes).astype("float")

        detections = []
        for _, row in px.iterrows():
            x1, y1, x2, y2, _, class_id = map(int, row)
            if class_list[class_id] == "person":
                detections.append([x1, y1, x2, y2])

        tracked_ids = tracker.update(detections)
        for x1, y1, x2, y2, id in tracked_ids:
            center = (x2, y2)

            if cv2.pointPolygonTest(np.array(exit_zone, np.int32), center, False) >= 0:
                going_out[id] = center
            if id in going_out and cv2.pointPolygonTest(np.array(entry_zone, np.int32), center, False) >= 0:
                if id not in counter_out_ids:
                    counter_out_ids.append(id)
                    st.session_state[out_key] = len(counter_out_ids)

            if cv2.pointPolygonTest(np.array(entry_zone, np.int32), center, False) >= 0:
                going_in[id] = center
            if id in going_in and cv2.pointPolygonTest(np.array(exit_zone, np.int32), center, False) >= 0:
                if id not in counter_in_ids:
                    counter_in_ids.append(id)
                    st.session_state[in_key] = len(counter_in_ids)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cvzone.putTextRect(frame, str(id), (x1, y1), 1, 1)

        cv2.polylines(frame, [np.array(entry_zone, np.int32)], True, (0, 255, 0), 2)
        cv2.polylines(frame, [np.array(exit_zone, np.int32)], True, (0, 0, 255), 2)

        video_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)
        out_display.metric("🔻 Out", st.session_state[out_key])
        in_display.metric("🔺 In", st.session_state[in_key])

    cap.release()

# Function for Route 3 (without session state, standalone tracking)
def run_people_counter_route3_custom(video_path):
    st.markdown(f"<h1 style='text-align:center;'> Bus Occupancy System - Route 3</h1>", unsafe_allow_html=True)
    st.markdown("---")

    col1, col2 = st.columns([3, 1])
    video_placeholder = col1.empty()
    with col2:
        st.subheader("📊 Counter")
        in_display = st.empty()
        st.markdown("---")

    area1 = [(259, 488), (281, 499), (371, 499), (303, 466)]

    tracker = Tracker_adv()  # Make sure this is correctly defined in tracker1
    counter = []

    cap = cv2.VideoCapture(video_path)
    count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.success("✅ Video processing completed!")
            break

        count += 1
        if count % 3 != 0:
            continue

        frame = cv2.resize(frame, (1020, 500))
        results = model.predict(frame, verbose=False)
        boxes = results[0].boxes.data
        px = pd.DataFrame(boxes).astype("float")

        person_detections = []
        for _, row in px.iterrows():
            x1, y1, x2, y2, _, class_id = map(int, row)
            class_name = class_list[class_id]
            if "person" in class_name:
                person_detections.append([x1, y1, x2, y2])

        bbox_idx = tracker.update(person_detections)

        for id, rect in bbox_idx.items():
            x3, y3, x4, y4 = rect
            cx, cy = x3, y4

            result = cv2.pointPolygonTest(np.array(area1, np.int32), (cx, cy), False)
            if result >= 0:
                cv2.circle(frame, (cx, cy), 6, (0, 255, 0), -1)
                cv2.rectangle(frame, (x3, y3), (x4, y4), (255, 0, 255), 2)
                cvzone.putTextRect(frame, f'{id}', (x3, y3), 1, 1)
                if id not in counter:
                    counter.append(id)

        p = len(counter)
        cvzone.putTextRect(frame, f'Counter:-{p}', (50, 60), 2, 2)
        cv2.polylines(frame, [np.array(area1, np.int32)], True, (255, 0, 0), 2)

        video_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)
        in_display.metric("🧍 Count", p)

    cap.release()

#function for route 2
def run_people_counter_route2_custom(video_path):
    st.markdown(f"<h1 style='text-align:center;'> Bus Occupancy System - {route}</h1>", unsafe_allow_html=True)
    st.markdown("---")

    col1, col2 = st.columns([3, 1])
    video_placeholder = col1.empty()
    with col2:
        st.subheader("📊 Counter")
        in_display = st.empty()
        st.markdown("---")

    tracker = Tracker_simp()
    area1 = [(494, 289), (505, 499), (578, 496), (530, 292)]
    going_in = {}
    counter2 = []

    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.success("✅ Video processing completed!")
            break

        frame = cv2.resize(frame, (1020, 500))
        results = model.predict(frame, verbose=False)
        boxes = results[0].boxes.data
        px = pd.DataFrame(boxes).astype("float")

        detections = []
        for _, row in px.iterrows():
            x1, y1, x2, y2, _, class_id = map(int, row)
            if class_list[class_id] == "person":
                detections.append([x1, y1, x2, y2])

        tracked_ids = tracker.update(detections)
        for x1, y1, x2, y2, id in tracked_ids:
            center = (x2, y2)

            if cv2.pointPolygonTest(np.array(area1, np.int32), center, False) >= 0:
                going_in[id] = center

            if id in going_in:
                result = cv2.pointPolygonTest(np.array(area1, np.int32), center, False)
                if result >= 0:
                    cv2.circle(frame, center, 7, (255, 0, 255), -1)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    cvzone.putTextRect(frame, f'{id}', (x1, y1), 1, 1)
                    if id not in counter2:
                        counter2.append(id)

        in_c = len(counter2)
        cvzone.putTextRect(frame, f'IN_C: {in_c}', (50, 60), 2, 2)
        cv2.polylines(frame, [np.array(area1, np.int32)], True, (0, 255, 0), 2)

        video_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)
        in_display.metric("🔺 In", in_c)

    cap.release()


# Route handling
if route == "Route 1":
    run_people_counter("p.mp4", "route1_in", "route1_out")
elif route == "Route 2":
    run_people_counter_route2_custom("new2.mp4")

elif route == "Route 3":
    run_people_counter_route3_custom("busfinal.mp4") 



