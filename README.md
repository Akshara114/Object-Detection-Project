# Object Detection Project

This project detects objects in images, videos, and live webcam streams.  
It draws bounding boxes around detected objects and shows their labels with confidence scores in real time.

It is built using YOLO and OpenCV for computer vision-based detection.

---

## Features

- Detects objects in images
- Processes videos frame by frame
- Real-time webcam detection
- Displays bounding boxes with labels and confidence scores
- Supports multiple object categories
- Stores detection history
- Simple and responsive web interface

---

## Tech Stack

- Python  
- Flask  
- YOLOv8 (Ultralytics)  
- OpenCV  
- SQLite  
- HTML, CSS, JavaScript  
---

## How to Run

### 1. Clone or open the project folder
```bash
cd project
**2. Create virtual environment**
python -m venv venv

# Activate on Windows
venv\Scripts\activate

# Activate on macOS/Linux
source venv/bin/activate
**3. Install dependencies**
pip install -r requirements.txt
**4. Run the application**
python app.py
**5. Open in browser**
http://localhost:5000
YOLO Models

You can switch models in the code depending on accuracy vs speed:

yolov8n.pt → fastest, light weight
yolov8s.pt → balanced
yolov8m.pt → better accuracy
yolov8l.pt → high accuracy
yolov8x.pt → best accuracy
**Requirements**
Python 3.9 or higher
Works on Windows, macOS, Linux
CPU supported (GPU recommended for faster video processing)
First run may download YOLO weights automatically

## Project Structure
