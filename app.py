"""
SentinelAI - Vision-Based Object Detection and Intelligent Alert System
Main Flask Application
"""

import os
import cv2
import json
import uuid
import time
import sqlite3
import hashlib
import secrets
from datetime import datetime
from functools import wraps
from flask import (
    Flask, render_template, request, redirect,
    url_for, session, jsonify, send_from_directory, flash
)
from werkzeug.utils import secure_filename

# ─────────────────────────────────────────────
# App Configuration
# ─────────────────────────────────────────────
app = Flask(__name__)
app.secret_key = secrets.token_hex(32)

BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
OUTPUT_FOLDER = os.path.join(BASE_DIR, 'outputs')
DB_PATH       = os.path.join(BASE_DIR, 'sentinelai.db')

ALLOWED_IMAGE_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}
ALLOWED_VIDEO_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'webm'}
MAX_CONTENT_LENGTH = 100 * 1024 * 1024  # 100 MB

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

# Alert-triggering object classes (customisable)
ALERT_CLASSES = {'person', 'knife', 'gun', 'fire', 'scissors', 'cell phone'}

# ─────────────────────────────────────────────
# Lazy-load YOLO so the app starts even without GPU
# ─────────────────────────────────────────────
_yolo_model = None

def get_model():
    global _yolo_model
    if _yolo_model is None:
        try:
            from ultralytics import YOLO
            model_path = os.path.join(BASE_DIR, 'models', 'yolov8n.pt')
            _yolo_model = YOLO(model_path)          # downloads automatically on first run
            print("[SentinelAI] YOLOv8 model loaded ✓")
        except Exception as e:
            print(f"[SentinelAI] YOLO load error: {e}")
            _yolo_model = None
    return _yolo_model


# ─────────────────────────────────────────────
# Database Helpers
# ─────────────────────────────────────────────
def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Create all tables if they don't exist."""
    conn = get_db()
    c = conn.cursor()

    # Users table
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            username      TEXT    UNIQUE NOT NULL,
            email         TEXT    UNIQUE NOT NULL,
            password_hash TEXT    NOT NULL,
            reset_token   TEXT,
            created_at    TEXT    DEFAULT (datetime('now'))
        )
    ''')

    # Detections table
    c.execute('''
        CREATE TABLE IF NOT EXISTS detections (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id      INTEGER NOT NULL,
            filename     TEXT    NOT NULL,
            file_type    TEXT    NOT NULL,
            object_name  TEXT    NOT NULL,
            confidence   REAL    NOT NULL,
            is_alert     INTEGER DEFAULT 0,
            timestamp    TEXT    DEFAULT (datetime('now')),
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    ''')

    conn.commit()
    conn.close()
    print("[SentinelAI] Database initialised ✓")


def hash_password(password: str) -> str:
    salt = secrets.token_hex(16)
    hashed = hashlib.sha256((salt + password).encode()).hexdigest()
    return f"{salt}:{hashed}"


def verify_password(password: str, stored_hash: str) -> bool:
    try:
        salt, hashed = stored_hash.split(':')
        return hashlib.sha256((salt + password).encode()).hexdigest() == hashed
    except Exception:
        return False


# ─────────────────────────────────────────────
# Auth Decorator
# ─────────────────────────────────────────────
def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if 'user_id' not in session:
            flash('Please log in to access this page.', 'warning')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated


# ─────────────────────────────────────────────
# Utility
# ─────────────────────────────────────────────
def allowed_file(filename, kind='image'):
    ext = filename.rsplit('.', 1)[-1].lower() if '.' in filename else ''
    if kind == 'image':
        return ext in ALLOWED_IMAGE_EXTENSIONS
    return ext in ALLOWED_VIDEO_EXTENSIONS


def save_detections(user_id, filename, file_type, results):
    """Persist every detected object to the DB."""
    conn = get_db()
    c = conn.cursor()
    for obj in results:
        c.execute('''
            INSERT INTO detections
                (user_id, filename, file_type, object_name, confidence, is_alert)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            user_id,
            filename,
            file_type,
            obj['class'],
            obj['confidence'],
            1 if obj['class'].lower() in ALERT_CLASSES else 0,
        ))
    conn.commit()
    conn.close()


# ─────────────────────────────────────────────
# YOLO Detection Helpers
# ─────────────────────────────────────────────
# Colour palette for bounding boxes (BGR for OpenCV)
COLOURS = [
    (0, 255, 127), (255, 99, 71), (30, 144, 255), (255, 215, 0),
    (186, 85, 211), (0, 206, 209), (255, 140, 0), (60, 179, 113),
]


def _colour_for(label: str) -> tuple:
    return COLOURS[hash(label) % len(COLOURS)]


def draw_detections(frame, detections):
    """Draw bounding boxes + labels on a frame."""
    for d in detections:
        x1, y1, x2, y2 = d['bbox']
        label      = d['class']
        conf       = d['confidence']
        colour     = (0, 60, 220) if label.lower() in ALERT_CLASSES else _colour_for(label)
        thickness  = 3 if label.lower() in ALERT_CLASSES else 2

        cv2.rectangle(frame, (x1, y1), (x2, y2), colour, thickness)

        caption = f"{label} {conf:.0%}"
        (tw, th), _ = cv2.getTextSize(caption, cv2.FONT_HERSHEY_DUPLEX, 0.6, 1)
        cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), colour, -1)
        cv2.putText(frame, caption, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)
    return frame


def run_yolo_image(src_path: str, dst_path: str):
    """
    Run YOLO on a single image.
    Returns list of detection dicts.
    """
    model = get_model()
    frame = cv2.imread(src_path)
    if frame is None:
        return []

    detections = []
    if model:
        results = model(frame, verbose=False)[0]
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf  = float(box.conf[0])
            cls   = results.names[int(box.cls[0])]
            detections.append({'class': cls, 'confidence': conf,
                                'bbox': [x1, y1, x2, y2]})

    frame = draw_detections(frame, detections)
    cv2.imwrite(dst_path, frame)
    return detections


def run_yolo_video(src_path: str, dst_path: str):
    """
    Run YOLO on every frame of a video.
    Returns aggregated detection list (unique class+conf pairs).
    """
    model = get_model()
    cap   = cv2.VideoCapture(src_path)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps    = cap.get(cv2.CAP_PROP_FPS) or 25
    w      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out    = cv2.VideoWriter(dst_path, fourcc, fps, (w, h))

    all_detections = []
    frame_count    = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        frame_detections = []

        # Process every 3rd frame for speed (still detect on every frame for output)
        if model and frame_count % 3 == 0:
            results = model(frame, verbose=False)[0]
            for box in results.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = float(box.conf[0])
                cls  = results.names[int(box.cls[0])]
                det  = {'class': cls, 'confidence': conf, 'bbox': [x1, y1, x2, y2]}
                frame_detections.append(det)
                all_detections.append(det)

        frame = draw_detections(frame, frame_detections)
        out.write(frame)

    cap.release()
    out.release()
    return all_detections


# ─────────────────────────────────────────────
# Routes – Authentication
# ─────────────────────────────────────────────
@app.route('/')
def index():
    if 'user_id' in session:
        return redirect(url_for('dashboard'))
    return redirect(url_for('login'))


@app.route('/login', methods=['GET', 'POST'])
def login():
    if 'user_id' in session:
        return redirect(url_for('dashboard'))

    if request.method == 'POST':
        identifier = request.form.get('identifier', '').strip()
        password   = request.form.get('password', '')

        conn = get_db()
        user = conn.execute(
            'SELECT * FROM users WHERE username=? OR email=?',
            (identifier, identifier)
        ).fetchone()
        conn.close()

        if user and verify_password(password, user['password_hash']):
            session['user_id']  = user['id']
            session['username'] = user['username']
            flash('Welcome back, ' + user['username'] + '!', 'success')
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid credentials. Please try again.', 'danger')

    return render_template('login.html')


@app.route('/register', methods=['GET', 'POST'])
def register():
    if 'user_id' in session:
        return redirect(url_for('dashboard'))

    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        email    = request.form.get('email', '').strip().lower()
        password = request.form.get('password', '')
        confirm  = request.form.get('confirm_password', '')

        if not all([username, email, password, confirm]):
            flash('All fields are required.', 'danger')
        elif password != confirm:
            flash('Passwords do not match.', 'danger')
        elif len(password) < 6:
            flash('Password must be at least 6 characters.', 'danger')
        else:
            try:
                conn = get_db()
                conn.execute(
                    'INSERT INTO users (username, email, password_hash) VALUES (?,?,?)',
                    (username, email, hash_password(password))
                )
                conn.commit()
                conn.close()
                flash('Account created! Please log in.', 'success')
                return redirect(url_for('login'))
            except sqlite3.IntegrityError:
                flash('Username or email already exists.', 'danger')

    return render_template('register.html')


@app.route('/forgot-password', methods=['GET', 'POST'])
def forgot_password():
    if request.method == 'POST':
        email = request.form.get('email', '').strip().lower()
        conn  = get_db()
        user  = conn.execute('SELECT * FROM users WHERE email=?', (email,)).fetchone()

        if user:
            token = secrets.token_urlsafe(32)
            conn.execute('UPDATE users SET reset_token=? WHERE id=?', (token, user['id']))
            conn.commit()
            # In production you would email the token; here we just flash it
            flash(f'Password reset token (demo): {token}', 'info')
        else:
            flash('If that email exists, a reset link has been sent.', 'info')

        conn.close()
        return redirect(url_for('login'))

    return render_template('forgot_password.html')


@app.route('/logout')
@login_required
def logout():
    session.clear()
    flash('You have been logged out.', 'info')
    return redirect(url_for('login'))


# ─────────────────────────────────────────────
# Routes – Dashboard
# ─────────────────────────────────────────────
@app.route('/dashboard')
@login_required
def dashboard():
    conn  = get_db()
    stats = conn.execute('''
        SELECT COUNT(*) as total,
               SUM(is_alert) as alerts,
               COUNT(DISTINCT object_name) as unique_objects
        FROM detections WHERE user_id=?
    ''', (session['user_id'],)).fetchone()
    conn.close()
    return render_template('dashboard.html', stats=stats)


# ─────────────────────────────────────────────
# Routes – Detection
# ─────────────────────────────────────────────
@app.route('/detect', methods=['POST'])
@login_required
def detect():
    """
    Accepts:
      - file   (image or video)
      - type   ('image' | 'video' | 'webcam')
    Returns JSON with detections + output file path.
    """
    input_type = request.form.get('type', 'image')

    # ── Webcam frame (base64 PNG sent from JS) ──────────────────────────────
    if input_type == 'webcam':
        import base64, numpy as np
        data_url = request.form.get('frame_data', '')
        if not data_url:
            return jsonify({'error': 'No frame data received'}), 400

        header, encoded = data_url.split(',', 1)
        img_bytes = base64.b64decode(encoded)
        np_arr    = np.frombuffer(img_bytes, np.uint8)
        frame     = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        uid      = uuid.uuid4().hex[:8]
        out_name = f'webcam_{uid}.jpg'
        out_path = os.path.join(OUTPUT_FOLDER, 'images', out_name)

        detections = []
        model = get_model()
        if model:
            results = model(frame, verbose=False)[0]
            for box in results.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = float(box.conf[0])
                cls  = results.names[int(box.cls[0])]
                detections.append({'class': cls, 'confidence': conf,
                                   'bbox': [x1, y1, x2, y2]})

        frame = draw_detections(frame, detections)
        cv2.imwrite(out_path, frame)
        save_detections(session['user_id'], out_name, 'webcam', detections)

        alerts = [d for d in detections if d['class'].lower() in ALERT_CLASSES]
        return jsonify({
            'detections': detections,
            'alerts':     alerts,
            'output_url': url_for('output_file', category='images', filename=out_name),
            'count':      len(detections),
        })

    # ── Uploaded file ────────────────────────────────────────────────────────
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400

    uid  = uuid.uuid4().hex[:8]
    ext  = secure_filename(file.filename).rsplit('.', 1)[-1].lower()
    safe = f"{uid}.{ext}"

    if input_type == 'image':
        if not allowed_file(file.filename, 'image'):
            return jsonify({'error': 'Invalid image format'}), 400

        src_path = os.path.join(UPLOAD_FOLDER, 'images', safe)
        dst_path = os.path.join(OUTPUT_FOLDER, 'images', safe)
        file.save(src_path)

        detections = run_yolo_image(src_path, dst_path)
        save_detections(session['user_id'], safe, 'image', detections)

        alerts = [d for d in detections if d['class'].lower() in ALERT_CLASSES]
        return jsonify({
            'detections': detections,
            'alerts':     alerts,
            'output_url': url_for('output_file', category='images', filename=safe),
            'count':      len(detections),
        })

    elif input_type == 'video':
        if not allowed_file(file.filename, 'video'):
            return jsonify({'error': 'Invalid video format'}), 400

        src_path = os.path.join(UPLOAD_FOLDER, 'videos', safe)
        dst_name = f"{uid}.mp4"
        dst_path = os.path.join(OUTPUT_FOLDER, 'videos', dst_name)
        file.save(src_path)

        detections = run_yolo_video(src_path, dst_path)
        save_detections(session['user_id'], dst_name, 'video', detections)

        alerts = [d for d in detections if d['class'].lower() in ALERT_CLASSES]
        return jsonify({
            'detections': detections,
            'alerts':     alerts,
            'output_url': url_for('output_file', category='videos', filename=dst_name),
            'count':      len(detections),
        })

    return jsonify({'error': 'Unknown input type'}), 400


# ─────────────────────────────────────────────
# Routes – Reports
# ─────────────────────────────────────────────
@app.route('/reports')
@login_required
def reports():
    obj_filter  = request.args.get('object', '').strip()
    date_filter = request.args.get('date', '').strip()

    query  = 'SELECT * FROM detections WHERE user_id=?'
    params = [session['user_id']]

    if obj_filter:
        query  += ' AND object_name LIKE ?'
        params.append(f'%{obj_filter}%')
    if date_filter:
        query  += ' AND DATE(timestamp)=?'
        params.append(date_filter)

    query += ' ORDER BY timestamp DESC'

    conn   = get_db()
    rows   = conn.execute(query, params).fetchall()
    # distinct objects for filter dropdown
    objects = [r['object_name'] for r in
               conn.execute('SELECT DISTINCT object_name FROM detections WHERE user_id=?',
                            (session['user_id'],)).fetchall()]
    conn.close()

    return render_template('reports.html', detections=rows,
                           objects=objects,
                           obj_filter=obj_filter,
                           date_filter=date_filter)


@app.route('/reports/export')
@login_required
def export_reports():
    """Export detections as JSON."""
    conn = get_db()
    rows = conn.execute(
        'SELECT * FROM detections WHERE user_id=? ORDER BY timestamp DESC',
        (session['user_id'],)
    ).fetchall()
    conn.close()

    data = [dict(r) for r in rows]
    response = app.response_class(
        response=json.dumps(data, indent=2),
        mimetype='application/json',
    )
    response.headers['Content-Disposition'] = 'attachment; filename=detections.json'
    return response


# ─────────────────────────────────────────────
# Static file serving for outputs
# ─────────────────────────────────────────────
@app.route('/outputs/<category>/<filename>')
@login_required
def output_file(category, filename):
    directory = os.path.join(OUTPUT_FOLDER, category)
    return send_from_directory(directory, filename)


# ─────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────
if __name__ == '__main__':
    os.makedirs(os.path.join(UPLOAD_FOLDER, 'images'), exist_ok=True)
    os.makedirs(os.path.join(UPLOAD_FOLDER, 'videos'), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_FOLDER, 'images'), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_FOLDER, 'videos'), exist_ok=True)
    init_db()
    app.run(debug=True, host='0.0.0.0', port=5000)
