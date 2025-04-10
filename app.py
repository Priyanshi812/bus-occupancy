APP.py
from flask import Flask, request, jsonify, render_template_string, send_from_directory
from werkzeug.utils import secure_filename
import os
import cv2
import numpy as np
from ultralytics import YOLO
import time

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
RESULTS_FOLDER = 'results'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

# Load YOLO model
try:
    model = YOLO('yolov8s.pt')
    print("✅ YOLO model loaded successfully")
except Exception as e:
    print(f"❌ Error loading YOLO model: {e}")
    model = None

# Define class for tracking objects
class Tracker:
    def __init__(self):
        self.center_points = {}
        # Keep track of ID count
        self.id_count = 0
        # Store tracked objects with their IDs
        self.objects = {}

    def update(self, objects_rect):
        # Objects boxes and ids
        objects_bbs_ids = []

        # Get center point of new object
        for rect in objects_rect:
            x, y, w, h = rect
            cx = (x + x + w) // 2
            cy = (y + y + h) // 2

            # Find out if that object was detected already
            same_object_detected = False
            for id, pt in self.center_points.items():
                dist = np.sqrt((cx - pt[0]) ** 2 + (cy - pt[1]) ** 2)

                if dist < 25:
                    self.center_points[id] = (cx, cy)
                    objects_bbs_ids.append([x, y, w, h, id])
                    same_object_detected = True
                    break

            # New object is detected we assign the ID to that object
            if same_object_detected is False:
                self.center_points[self.id_count] = (cx, cy)
                objects_bbs_ids.append([x, y, w, h, self.id_count])
                self.id_count += 1

        # Clean the dictionary by center points to remove IDs not used anymore
        new_center_points = {}
        for obj_bb_id in objects_bbs_ids:
            _, _, _, _, object_id = obj_bb_id
            center = self.center_points[object_id]
            new_center_points[object_id] = center

        # Update dictionary with IDs not used removed
        self.center_points = new_center_points.copy()
        return objects_bbs_ids

# Define the area for counting
def is_in_counting_area(center, counting_area):
    # Simplified point in polygon check
    result = cv2.pointPolygonTest(np.array(counting_area), center, False)
    return result >= 0

@app.route('/')
def index():
    return render_template_string('''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Crowd Detection</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                max-width: 800px;
                margin: 0 auto;
                padding: 20px;
            }
            h1 {
                color: #333;
            }
            .upload-form {
                border: 2px dashed #ccc;
                padding: 20px;
                border-radius: 5px;
                margin-bottom: 20px;
            }
            .submit-btn {
                background-color: #4CAF50;
                color: white;
                padding: 10px 15px;
                border: none;
                border-radius: 4px;
                cursor: pointer;
                font-size: 16px;
            }
            .loading {
                display: none;
                margin-top: 20px;
            }
            #result {
                margin-top: 20px;
                padding: 10px;
                border: 1px solid #ddd;
                border-radius: 4px;
                display: none;
            }
        </style>
    </head>
    <body>
        <h1>Crowd Detection System</h1>
        <div class="upload-form">
            <h2>Upload Video</h2>
            <form id="uploadForm" method="POST" action="/upload" enctype="multipart/form-data">
                <input type="file" name="video" accept="video/*" required>
                <br><br>
                <button type="submit" class="submit-btn">Analyze Video</button>
            </form>
            <div id="loading" class="loading">
                <p>Processing video... This may take a few minutes.</p>
                <progress id="progressBar" value="0" max="100" style="width: 100%;"></progress>
            </div>
        </div>
        <div id="result"></div>

        <script>
            document.getElementById('uploadForm').addEventListener('submit', function(e) {
                e.preventDefault();
                
                document.getElementById('loading').style.display = 'block';
                document.getElementById('result').style.display = 'none';
                
                const formData = new FormData(this);
                
                fetch('/upload', {
                    method: 'POST',
                    body: formData
                })
                .then(response => response.json())
                .then(data => {
                    document.getElementById('loading').style.display = 'none';
                    
                    const resultDiv = document.getElementById('result');
                    resultDiv.style.display = 'block';
                    
                    if (data.video_path) {
                        resultDiv.innerHTML = `
                            <h2>Results:</h2>
                            <p>People detected: ${data.people_count}</p>
                            <p>Average crowd density: ${data.avg_crowd_density.toFixed(2)} people per frame</p>
                            <video width="100%" controls>
                                <source src="${data.video_path}" type="video/mp4">
                                Your browser does not support the video tag.
                            </video>
                        `;
                    } else {
                        resultDiv.innerHTML = `<p>Error: ${data.message || 'Unknown error'}</p>`;
                    }
                })
                .catch(error => {
                    document.getElementById('loading').style.display = 'none';
                    document.getElementById('result').style.display = 'block';
                    document.getElementById('result').innerHTML = `<p>Error: ${error.message}</p>`;
                });
            });
        </script>
    </body>
    </html>
    ''')

@app.route('/upload', methods=['POST'])
def upload_video():
    print("🔹 Received request")

    if 'video' not in request.files:
        print("❌ No file part")
        return jsonify({"message": "No file part"}), 400

    file = request.files['video']
    if file.filename == '':
        print("❌ File name is empty")
        return jsonify({"message": "No selected file"}), 400

    filename = secure_filename(file.filename)
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    print(f"📁 Saving file: {filename}")
    file.save(file_path)

    try:
        # Process the video
        result = process_video(file_path)
        return jsonify(result)
    except Exception as e:
        print("❌ Error during processing:", e)
        return jsonify({"message": f"Error processing video: {str(e)}"}), 500

def process_video(video_path):
    if model is None:
        raise Exception("YOLO model not loaded properly")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise Exception(f"Cannot open video file: {video_path}")
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Define the output file path
    output_filename = f"result_{os.path.basename(video_path)}"
    output_path = os.path.join(RESULTS_FOLDER, output_filename)
    
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Initialize Tracker
    tracker = Tracker()
    
    # Define counting area (adjust as needed for your video)
    counting_area = [
        (int(width * 0.3), int(height * 0.3)), 
        (int(width * 0.7), int(height * 0.3)), 
        (int(width * 0.7), int(height * 0.7)), 
        (int(width * 0.3), int(height * 0.7))
    ]
    
    frame_count = 0
    total_people_count = 0
    people_counts = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        
        # Run detection on current frame
        results = model(frame, classes=[0])  # 0 is the class index for 'person'
        
        # Extract detection boxes for 'person' class
        detections = []
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = box.conf[0].item()
                cls = int(box.cls[0].item())
                
                if cls == 0 and conf > 0.3:  # Person class with confidence threshold
                    x, y, w, h = int(x1), int(y1), int(x2 - x1), int(y2 - y1)
                    detections.append([x, y, w, h])
        
        # Update tracker
        if detections:
            tracked_objects = tracker.update(detections)
            
            # Count people in the defined area
            people_in_area = 0
            
            for obj in tracked_objects:
                x, y, w, h, id = obj
                cx, cy = x + w // 2, y + h // 2
                
                # Draw bounding box
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
                # Draw ID
                cv2.putText(frame, f"ID: {id}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                # Check if person is in counting area
                if is_in_counting_area((cx, cy), counting_area):
                    people_in_area += 1
                    cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
                else:
                    cv2.circle(frame, (cx, cy), 5, (255, 0, 0), -1)
            
            # Draw counting area
            cv2.polylines(frame, [np.array(counting_area)], True, (0, 255, 255), 2)
            
            # Add count to frame
            cv2.putText(frame, f"People Count: {people_in_area}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            people_counts.append(people_in_area)
            
        # Write the frame to output video
        out.write(frame)
    
    # Release resources
    cap.release()
    out.release()
    
    # Calculate statistics
    avg_people_count = sum(people_counts) / len(people_counts) if people_counts else 0
    max_people_count = max(people_counts) if people_counts else 0
    
    # Create public URL for the video result
    video_url = f"/view_result/{output_filename}"
    
    return {
        "message": "Video processed successfully",
        "video_path": video_url,
        "people_count": max_people_count,
        "avg_crowd_density": avg_people_count,
        "frames_processed": frame_count
    }

@app.route('/view_result/<filename>')
def view_result(filename):
    return send_from_directory(RESULTS_FOLDER, filename)

if __name__ == '__main__':
    if model is None:
        print("⚠️ Warning: YOLO model failed to load. Application will not function correctly.")
    app.run(debug=True)