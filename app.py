from flask import Flask, render_template, Response, jsonify
import cv2
import numpy as np
from keras.models import load_model
from PIL import Image, ImageOps
import threading
import time
import base64

app = Flask(__name__)

# Global variables
camera = None
current_prediction = {"class": "Unknown", "confidence": 0.0, "color": "gray"}
model = None
class_names = []
detection_active = False

# Initialize the model and labels
def initialize_model():
    global model, class_names
    try:
        # Load the model
        model = load_model("keras_Model.h5", compile=False)
        
        # Load the labels
        with open("labels.txt", "r") as f:
            class_names = [line.strip() for line in f.readlines()]
        
        print("Model and labels loaded successfully!")
        return True
    except Exception as e:
        print(f"Error loading model: {e}")
        return False

# Preprocess image for prediction
def preprocess_image(frame):
    # Convert BGR to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Convert to PIL Image
    pil_image = Image.fromarray(rgb_frame)
    
    # Resize and crop to 224x224
    size = (224, 224)
    pil_image = ImageOps.fit(pil_image, size, Image.Resampling.LANCZOS)
    
    # Convert back to numpy array
    image_array = np.asarray(pil_image)
    
    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
    
    # Create batch dimension
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_image_array
    
    return data

# Predict traffic light
def predict_traffic_light(frame):
    global model, class_names, current_prediction
    
    if model is None:
        return
    
    try:
        # Preprocess the frame
        processed_frame = preprocess_image(frame)
        
        # Make prediction
        prediction = model.predict(processed_frame, verbose=0)
        index = np.argmax(prediction)
        
        if index < len(class_names):
            class_name = class_names[index]
            confidence_score = float(prediction[0][index])
            
            # Extract class name (remove index prefix)
            clean_class_name = class_name.split(' ', 1)[1] if ' ' in class_name else class_name
            
            # Determine background color
            color_map = {
                "Stop": "#dc3545",  # Red
                "Be Ready": "#ffc107",  # Yellow
                "Go": "#28a745"  # Green
            }
            
            current_prediction = {
                "class": clean_class_name,
                "confidence": confidence_score,
                "color": color_map.get(clean_class_name, "#6c757d")
            }
    except Exception as e:
        print(f"Prediction error: {e}")

# Generate video frames
def generate_frames():
    global camera, detection_active
    
    camera = cv2.VideoCapture(0)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    while True:
        success, frame = camera.read()
        if not success:
            break
        
        # Perform prediction if detection is active
        if detection_active and model is not None:
            predict_traffic_light(frame)
        
        # Add prediction overlay
        if current_prediction["class"] != "Unknown":
            # Add semi-transparent overlay
            overlay = frame.copy()
            cv2.rectangle(overlay, (10, 10), (300, 80), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
            
            # Add text
            cv2.putText(frame, f"Class: {current_prediction['class']}", 
                       (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"Confidence: {current_prediction['confidence']:.2f}", 
                       (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Encode frame
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def dashboard():
    return render_template('dashboard.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start_detection')
def start_detection():
    global detection_active
    detection_active = True
    return jsonify({"status": "Detection started"})

@app.route('/stop_detection')
def stop_detection():
    global detection_active
    detection_active = False
    return jsonify({"status": "Detection stopped"})

@app.route('/get_prediction')
def get_prediction():
    return jsonify(current_prediction)

@app.route('/toggle_detection')
def toggle_detection():
    global detection_active
    detection_active = not detection_active
    status = "started" if detection_active else "stopped"
    return jsonify({"status": f"Detection {status}", "active": detection_active})

if __name__ == '__main__':
    print("Initializing Traffic Light Detection System...")
    
    if initialize_model():
        print("Starting Flask application...")
        app.run(debug=True, threaded=True)
    else:
        print("Failed to initialize model. Please check your model files.")