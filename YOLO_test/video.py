# Import necessary libraries
from ultralytics import YOLO
import cv2

# Load the YOLOv8 model
model = YOLO("yolov8n.pt")  # Use a lightweight model for better speed

# Open a connection to the laptop camera
camera = cv2.VideoCapture(0)  # 0 is the default camera (change if needed)

# Check if the camera is accessible
if not camera.isOpened():
    print("Error: Unable to access the camera.")
    exit()

print("Press 'q' to quit the video feed.")

# Loop to process each frame
while True:
    ret, frame = camera.read()
    if not ret:
        print("Error: Unable to capture video frame.")
        break

    # resize the frame
    resized_frame = cv2.resize(frame, (1040, 1040))

    # Run YOLO inference on the frame
    results = model(frame)

    # Annotate the frame with YOLO's predictions
    annotated_frame = results[0].plot()

    # Display the annotated frame
    cv2.imshow("YOLOv8 - Laptop Camera", annotated_frame)

    # Press 'q' to quit the video feed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all OpenCV windows
camera.release()
cv2.destroyAllWindows()
