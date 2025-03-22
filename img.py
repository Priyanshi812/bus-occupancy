import cv2
import os

# Define the output directory
output_dir = "/Users/dhruvibhalodiya/Desktop/BUS OCCUPANCY SYSTEM/images"

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# Video file path
video_path = "new2.mp4"

# Open the video file
cap = cv2.VideoCapture(video_path)

# Check if the video opened successfully
if not cap.isOpened():
    print("Error: Unable to open video file.")
    exit(1)

# Frame processing parameters
frame_interval = 3  # Process every 3rd frame
max_frames = 200  # Maximum frames to save
frame_count = 0
saved_count = 0

while saved_count < max_frames:
    ret, frame = cap.read()
    if not ret:
        print("End of video or cannot read frame.")
        break

    # Process every 'frame_interval' frame
    if frame_count % frame_interval == 0:
        # Resize the frame
        frame_resized = cv2.resize(frame, (1080, 500))

        # Save the frame as an image
        image_path = os.path.join(output_dir, f"person_{saved_count}.jpg")
        cv2.imwrite(image_path, frame_resized)

        # Display the frame
        cv2.imshow("Captured Frame", frame_resized)

        saved_count += 1

    frame_count += 1

    # Exit if 'Esc' key is pressed
    if cv2.waitKey(5) & 0xFF == 27:
        break

# Release resources
cap.release()
cv2.destroyAllWindows()

print(f"Frames saved successfully in: {output_dir}")

