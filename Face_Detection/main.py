from ultralytics import YOLO
import cv2

# Load the trained YOLO model
model = YOLO("runs/detect/train5/weights/best.pt")

# Open a connection to the laptop camera
camera = cv2.VideoCapture(0)  # Use 0 for the default camera

# Check if the camera is accessible
if not camera.isOpened():
    print("Error: Unable to access the camera.")
    exit()

print("Press 'q' to quit the video feed.")

# Loop to process each frame
while True:
    ret, frame = camera.read()  # Capture frame from the camera
    if not ret:
        print("Error: Unable to capture video frame.")
        break

    mirrored_frame = cv2.flip(frame, 1)

    # Run YOLO inference on the mirrored frame
    results = model.predict(source=mirrored_frame, verbose=False)

    # Annotate the frame with the predictions
    annotated_frame = results[0].plot()

    # Display the annotated frame
    cv2.imshow("YOLOv8 - Live Camera Feed", annotated_frame)

    # Press 'q' to quit the video feed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all OpenCV windows
camera.release()
cv2.destroyAllWindows()