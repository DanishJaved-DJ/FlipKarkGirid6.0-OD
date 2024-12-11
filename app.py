from flask import Flask, render_template, Response, request, jsonify
from ultralytics import YOLO
import cv2
import sqlite3
from datetime import datetime, timedelta

app = Flask(__name__)

# Load the YOLOv8 model
model = YOLO('model.pt')  # Replace with your trained model

# Initialize the webcam
cap = cv2.VideoCapture(0)  # 0 for default webcam
if not cap.isOpened():
    raise Exception("Could not open webcam.")

paused = False

# Connect to SQLite database
conn = sqlite3.connect('detections.db', check_same_thread=False)
cursor = conn.cursor()

# Create the table if it doesn't exist
cursor.execute('''
CREATE TABLE IF NOT EXISTS detections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    product_detected TEXT,
    quantity INTEGER,
    timestamp TEXT
)
''')
conn.commit()

def get_last_saved_time(product_detected):
    """Retrieve the most recent timestamp for a specific product."""
    cursor.execute('''
        SELECT MAX(timestamp) FROM detections WHERE product_detected = ?
    ''', (product_detected,))
    result = cursor.fetchone()
    if result[0]:
        return datetime.strptime(result[0], '%Y-%m-%d %H:%M:%S')
    return None

def save_to_database(product_detected, quantity):
    """Save detection results to the database if the timestamp difference is at least 20 seconds."""
    last_timestamp = get_last_saved_time(product_detected)
    current_time = datetime.now()
    if not last_timestamp or (current_time - last_timestamp) >= timedelta(seconds=20):
        timestamp = current_time.strftime('%Y-%m-%d %H:%M:%S')
        cursor.execute('''
            INSERT INTO detections (product_detected, quantity, timestamp)
            VALUES (?, ?, ?)
        ''', (product_detected, quantity, timestamp))
        conn.commit()

def generate_frames():
    global paused
    while True:
        if paused:
            continue

        success, frame = cap.read()
        if not success:
            break

        # Perform object detection
        results = model.predict(source=frame)
        detections = results[0].boxes

        # Parse detections
        detected_items = {}
        for box in detections:
            cls = int(box.cls[0])  # Class ID
            name = model.names[cls]  # Class name
            detected_items[name] = detected_items.get(name, 0) + 1

        # Save detections to the database
        for product, quantity in detected_items.items():
            save_to_database(product, quantity)

        # Annotate the frame
        annotated_frame = results[0].plot()

        # Encode the frame as JPEG
        ret, buffer = cv2.imencode('.jpg', annotated_frame)
        frame = buffer.tobytes()

        # Yield the frame
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_data')
def get_data():
    cursor.execute('SELECT * FROM detections ORDER BY timestamp DESC')
    rows = cursor.fetchall()
    return jsonify(rows)

@app.route('/control', methods=['POST'])
def control():
    global paused
    data = request.get_json()
    command = data.get('command')

    if command == 'pause':
        paused = True
        return jsonify({'status': 'success'})
    elif command == 'play':
        paused = False
        return jsonify({'status': 'success'})
    elif command == 'end':
        cap.release()
        return jsonify({'status': 'success'})
    else:
        return jsonify({'status': 'error', 'error': 'Invalid command'})

if __name__ == '__main__':
    app.run(debug=True)
# import os

# if __name__ == '__main__':
#     port = int(os.environ.get('PORT', 5000))  # Use PORT from environment variables or default to 5000
#     app.run(debug=False, host='0.0.0.0', port=port)
