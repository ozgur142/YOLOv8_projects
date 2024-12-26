
# Face Detection with YOLO

This project demonstrates how to train a YOLO model to detect faces and use it for real-time applications such as camera-based detection.

## Features
- Train a YOLO model on a custom face detection dataset.
- Detect faces in real time using a camera or video files.

## Installation
### **Step 1: Clone the repository**
```bash
git clone git@github.com:ozgur142/YOLOv8_projects.git
cd YOLOv8_projects/Face_Detection
```
### **Step 2: Create a virtual environment (recommended)**
#### 1. Create the virtual environment:
```bash
python3 -m venv myenv
```

#### 2. Activate the virtual environment:
- On Linux/Mac:
```bash
source myenv/bin/activate
```

- On Windows:
```bash
myenv\Scripts\activate
```

### **Step 3: Install dependencies**
```bash
pip install -r requirements.txt
```

### **Step 4: Download the dataset**
#### 1. Download the dataset:
```bash
kaggle datasets download -d fareselmenshawii/face-detection-dataset
```
#### 2. Extract the dataset:
```bash
python3 extract.py
```

## Usage
### Training the Model
Verify the config.yaml file.
Run the train.py script to train the YOLO model on the dataset:
```bash
python3 train.py
```

### Testing the Model
Run the main.py script to test the model on a live camera feed:
```bash
python3 main.py
```

## Additional Notes
- Make sure to update the config.yaml file to point to your dataset paths.
- For better performance, consider training for more epochs or using a more powerful YOLO model variant.

## License

[MIT](https://choosealicense.com/licenses/mit/)