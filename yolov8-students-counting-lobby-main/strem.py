import cv2
import numpy as np
from ultralytics import YOLO
import streamlit as st
import os

# Streamlit page config - must be first command
st.set_page_config(
    layout="wide",
    page_title="People Counter",
    page_icon="👥"
)

# Sidebar controls
with st.sidebar:
    st.title("Settings")
    video_path = st.text_input("Video path", "new2.mp4")
    model_path = st.text_input("Model path", "yolov8s.pt")
    show_zones = st.checkbox("Show detection zones", True)
    detection_confidence = st.slider("Detection confidence threshold", 0.1, 1.0, 0.5)
    pause_button = st.button("⏸️ Pause")
    resume_button = st.button("▶️ Resume")
    
    st.markdown("---")
    st.markdown("### Zone Coordinates")
    st.write("Entry Zone:", [(494, 289), (505, 499), (578, 496), (530, 292)])
    st.write("Exit Zone:", [(548, 290), (600, 496), (637, 493), (574, 288)])

# Check if the video file exists
if not os.path.exists(video_path):
    st.error(f"Error: The video file does not exist at {video_path}")
    st.stop()

# Open the video file
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    st.error(f"Error: Failed to open the video at {video_path}")
    st.stop()

# Load YOLO model
try:
    model = YOLO(model_path)
except Exception as e:
    st.error(f"Error loading model: {str(e)}")
    st.stop()

# Define entry and exit zones (polygons)
area_entry = [(494, 289), (505, 499), (578, 496), (530, 292)]  # Entry area
area_exit = [(548, 290), (600, 496), (637, 493), (574, 288)]   # Exit area

# Initialize session state for tracking
if 'entry_count' not in st.session_state:
    st.session_state.entry_count = 0
    st.session_state.exit_count = 0
    st.session_state.people_entered = set()
    st.session_state.people_exited = set()
    st.session_state.paused = False

# Main app layout
st.title("👥 Real-time People Counter")
st.markdown("---")

# Create columns for counters and video
col1, col2, col3 = st.columns([1, 1, 3])

with col1:
    st.metric("People Entering", st.session_state.entry_count, delta_color="off")

with col2:
    st.metric("People Exiting", st.session_state.exit_count, delta_color="off")

with col3:
    frame_placeholder = st.empty()
    st.markdown("---")
    status_text = st.empty()

# Main loop to process video
while cap.isOpened():
    if not st.session_state.paused:
        ret, frame = cap.read()
        if not ret:
            status_text.warning("Video ended or cannot read frame")
            break

        # Resize the frame for better performance
        frame = cv2.resize(frame, (1020, 500))

        # Perform object detection using YOLO
        results = model.predict(frame, conf=detection_confidence)
        boxes = results[0].boxes.data
        detected_people = []

        for row in boxes:
            x1, y1, x2, y2, conf, cls = map(int, row)
            if cls == 0:  # Class 0 is 'person'
                detected_people.append([x1, y1, x2, y2])
                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

        # Loop through detected people
        for bbox in detected_people:
            x1, y1, x2, y2 = bbox
            centroid = ((x1 + x2) // 2, (y1 + y2) // 2)
            
            # Draw centroid
            cv2.circle(frame, centroid, 5, (0, 255, 255), -1)

            # Check if the person is entering or exiting
            entering = cv2.pointPolygonTest(np.array(area_entry, np.int32), centroid, False)
            exiting = cv2.pointPolygonTest(np.array(area_exit, np.int32), centroid, False)

            # If person is inside the entry area
            if entering >= 0 and centroid not in st.session_state.people_entered and centroid not in st.session_state.people_exited:
                st.session_state.people_entered.add(centroid)
                st.session_state.entry_count += 1
                # Visual feedback for entry
                cv2.putText(frame, "ENTER", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            # If person is inside the exit area
            if exiting >= 0 and centroid not in st.session_state.people_exited:
                st.session_state.people_exited.add(centroid)
                st.session_state.exit_count += 1
                # Visual feedback for exit
                cv2.putText(frame, "EXIT", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        # Draw the entry and exit zones if enabled
        if show_zones:
            cv2.polylines(frame, [np.array(area_entry, np.int32)], True, (0, 255, 0), 2)
            cv2.putText(frame, "Entry Zone", (area_entry[0][0], area_entry[0][1]-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.polylines(frame, [np.array(area_exit, np.int32)], True, (0, 0, 255), 2)
            cv2.putText(frame, "Exit Zone", (area_exit[0][0], area_exit[0][1]-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Convert BGR to RGB for Streamlit display
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Show the frame in the Streamlit app
        frame_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
       
    # Handle pause/resume
    if pause_button:
        st.session_state.paused = True
    if resume_button:
        st.session_state.paused = False

    # Add a small delay to prevent high CPU usage
    cv2.waitKey(10)

cap.release()
cv2.destroyAllWindows()