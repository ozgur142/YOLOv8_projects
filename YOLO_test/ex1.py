from ultralytics import YOLO
import cv2

# Load the pre-trained YOLOv8 model (coco dataset)
model = YOLO('yolov8n.pt')  # 'yolov8n.pt' is a lightweight model trained on 80 classes

# Perform inference on an example image
image_path = 'img/img1.JPG'  # Replace with the path to your image
results = model(image_path)  # Detect objects in the image

# Extract results
print("Detection Results:")
for result in results[0].boxes:
    cls_id = int(result.cls[0])  # Class ID
    confidence = result.conf[0]  # Confidence score
    box = result.xyxy[0]  # Bounding box coordinates (x1, y1, x2, y2)
    print(f"Class ID: {cls_id}, Confidence: {confidence:.2f}, Box: {box.tolist()}")

# Visualize the results using OpenCV
result_image = results[0].plot()  # Draw bounding boxes on the image

# Resize the image to fit your screen (optional)
max_width = 800  # Set the desired maximum width for display
height, width, _ = result_image.shape
if width > max_width:
    scale = max_width / width
    result_image = cv2.resize(result_image, (int(width * scale), int(height * scale)))



# Display the resized image
cv2.imshow('YOLOv8 Detection', result_image)


while True:
    key = cv2.waitKey(1)  # Wait for 1 ms between checks
    if key == 27 or cv2.getWindowProperty('YOLOv8 Detection', cv2.WND_PROP_VISIBLE) < 1:
        # Exit if 'ESC' is pressed or the window is closed
        break

# Close all OpenCV windows
cv2.destroyAllWindows()


# Save the result (optional)
cv2.imwrite('output.jpg', result_image)  # Save the image with detections