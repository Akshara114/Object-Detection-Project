<<<<<<< HEAD
# SentinelAI — Vision-Based Object Detection & Intelligent Alert System

> **Stack:** Python · Flask · YOLOv8 · OpenCV · SQLite · HTML/CSS/JS

---

## Project Structure

```
sentinelai/
├── app.py                  ← Flask application (all routes + logic)
├── init_db.py              ← Standalone DB setup script
├── generate_sound.py       ← Generates alert beep (optional)
├── requirements.txt
├── sentinelai.db           ← SQLite database (auto-created)
│
├── models/
│   └── yolov8n.pt          ← Downloaded automatically on first run
│
├── uploads/
│   ├── images/             ← Uploaded source images
│   └── videos/             ← Uploaded source videos
│
├── outputs/
│   ├── images/             ← Processed images with bounding boxes
│   └── videos/             ← Processed videos with bounding boxes
│
├── static/
│   ├── css/style.css
│   ├── js/
│   │   ├── main.js
│   │   └── dashboard.js
│   └── sounds/alert.mp3    ← (optional) alert beep
│
└── templates/
    ├── base.html
    ├── login.html
    ├── register.html
    ├── forgot_password.html
    ├── dashboard.html
    └── reports.html
```

---

## Quick Start

### 1. Clone / unzip the project
```bash
cd sentinelai
```

### 2. Create a virtual environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

> **Note:** `ultralytics` will download **yolov8n.pt** (~6 MB) automatically on first detection run if it is not present in `models/`.

### 4. (Optional) Generate alert sound
```bash
python generate_sound.py
# Then convert to MP3:
ffmpeg -i static/sounds/alert.wav static/sounds/alert.mp3
```

### 5. Run the application
```bash
python app.py
```

Open your browser at **http://localhost:5000**

---

## Features

| Feature | Details |
|---|---|
| **Authentication** | Register / Login / Forgot Password with bcrypt-style hashing |
| **Image Detection** | Upload PNG/JPG/BMP/WEBP → YOLOv8 inference → bounding-box overlay |
| **Video Detection** | Upload MP4/AVI/MOV → frame-by-frame detection → annotated video output |
| **Webcam Detection** | Live camera → capture frame → instant detection |
| **Alert System** | Triggers on `person`, `knife`, `gun`, `fire`, `scissors`, `cell phone` (configurable) |
| **Reports** | Full detection history, filterable by object or date, JSON export |
| **Session Security** | Session-based auth with `@login_required` decorator |
| **Responsive UI** | Works on desktop and mobile |

---

## Customising Alert Classes

Edit `ALERT_CLASSES` in `app.py`:

```python
ALERT_CLASSES = {'person', 'knife', 'gun', 'fire', 'scissors', 'cell phone'}
```

You can use any COCO class name (80 classes). Full list: https://docs.ultralytics.com/datasets/detect/coco/

---

## Switching YOLO Model

| Model | Speed | Accuracy | Size |
|---|---|---|---|
| `yolov8n.pt` | Fastest | Lower | 6 MB |
| `yolov8s.pt` | Fast | Good | 22 MB |
| `yolov8m.pt` | Medium | Better | 50 MB |
| `yolov8l.pt` | Slow | High | 84 MB |
| `yolov8x.pt` | Slowest | Highest | 131 MB |

Change the model in `app.py → get_model()`:
```python
_yolo_model = YOLO('yolov8s.pt')  # or full path
```

---

## Environment Notes

- **Python 3.9+** required
- Works on Windows, macOS, Linux
- GPU (CUDA) is auto-detected by Ultralytics for faster inference
- CPU-only inference is supported (slower for video)

---

## Security Notes

- Passwords are salted + SHA-256 hashed (replace with `bcrypt` for production)
- Uploaded files are sanitised with `werkzeug.secure_filename`
- Sessions use a random 32-byte secret key
- File size limit: 100 MB
- Only authenticated users can access detection & reports routes
=======
# Object-Detection-Project
A deep learning–based Object Detection project that can identify and locate multiple objects in images, videos, and real-time webcam streams. The system uses Computer Vision and models like YOLO with OpenCV to draw bounding boxes around detected objects and display their labels with confidence scores.
>>>>>>> dab6f321dfb805326267e4eccd804d00e26932b7
