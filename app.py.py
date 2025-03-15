from flask import Flask, render_template, Response
import cv2
import time

app = Flask(__name__)

# Initialize video capture
cap = cv2.VideoCapture('busfinal.mp4')

def generate_frames():
    while True:
        success, frame = cap.read()
        if not success:
            print("🚨 Error: Couldn't read frame from video.")
            break
        else:
            print("✅ Frame captured successfully")  # Debugging log
            
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/')
def index():
    return render_template('index.html')  # Make sure 'index.html' is inside 'templates' folder

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)
