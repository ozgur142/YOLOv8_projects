# kaggle datasets download -d fareselmenshawii/face-detection-dataset

import zipfile
import os
import cv2

zip_file_path = "face-detection-dataset.zip"
extract_folder = "Face_Dataset"

# Extract the dataset
with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    zip_ref.extractall(extract_folder)

delete_path = "Face_Dataset/labels2"
# Delete unnecceary files
for file in os.listdir(delete_path):
    file_path = os.path.join(delete_path, file)
    if os.path.isfile(file_path):
        os.remove(file_path)  # Delete file
    elif os.path.isdir(file_path):
        os.rmdir(file_path)  # Delete empty subdirectory (if applicable)

# Delete the now-empty folder
os.rmdir(delete_path)

print(f"Dataset extracted to {extract_folder}")



# Paths to the dataset
images_path = "Face_Dataset/images/train"
labels_path = "Face_Dataset/labels/train"

class_names = ["face"]

def draw_boxes(image, label_file):
    height, width, _ = image.shape

    # Open the label file and draw each bounding box
    with open(label_file, "r") as f:
        for line in f.readlines():
            class_id, x_center, y_center, bbox_width, bbox_height = map(float, line.split())
            
            # Convert YOLO normalized coordinates to pixel values
            x_center *= width
            y_center *= height
            bbox_width *= width
            bbox_height *= height

            # Calculate the top-left and bottom-right corners of the bounding box
            x1 = int(x_center - bbox_width / 2)
            y1 = int(y_center - bbox_height / 2)
            x2 = int(x_center + bbox_width / 2)
            y2 = int(y_center + bbox_height / 2)

            # Draw the bounding box
            color = (0, 255, 0)  # Green box for visualization
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

            # Add the class label
            label = class_names[int(class_id)] if int(class_id) < len(class_names) else f"class_{int(class_id)}"
            cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return image

# Iterate through the images and visualize the labels
for image_file in os.listdir(images_path)[:5]:
    if image_file.endswith(".jpg"):  # Adjust if your images have a different extension
        image_path = os.path.join(images_path, image_file)
        label_file = os.path.join(labels_path, image_file.replace(".jpg", ".txt"))

        # Check if the label file exists for the image
        if not os.path.exists(label_file):
            print(f"Warning: Label file not found for {image_file}")
            continue

        # Load the image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Unable to load image {image_path}")
            continue

        # Draw bounding boxes on the image
        annotated_image = draw_boxes(image, label_file)

        # Display the image with bounding boxes
        cv2.imshow("Annotated Image", annotated_image)

        if cv2.waitKey(0) & 0xFF == ord('q'):
            break

# Close all OpenCV windows
cv2.destroyAllWindows()