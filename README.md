# Traffic Light Detection Flask App - Setup Instructions

## 📁 Folder Structure
Create the following folder structure:

```
traffic_light_app/
│
├── app.py                    # Main Flask application
├── keras_Model.h5           # Your trained model file
├── labels.txt               # Your labels file
├── templates/
│   └── dashboard.html       # Dashboard template
├── static/                  # (Optional: for additional CSS/JS)
└── requirements.txt         # Python dependencies
```

## 🛠️ Installation Steps

### 1. Create Project Directory
```bash
mkdir traffic_light_app
cd traffic_light_app
```

### 2. Create Virtual Environment (Recommended)
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 3. Install Required Packages
Create `requirements.txt` file:
```
flask==2.3.3
opencv-python==4.8.1.78
tensorflow==2.13.0
keras==2.13.1
pillow==10.0.0
numpy==1.24.3
```

Install dependencies:
```bash
pip install -r requirements.txt
```

### 4. Create Templates Folder
```bash
mkdir templates
```

### 5. Place Your Files
- Copy your `keras_Model.h5` to the root directory
- Copy your `labels.txt` to the root directory
- Save the Flask app code as `app.py`
- Save the HTML template as `templates/dashboard.html`

### 6. Update Labels File Format (Important!)
Make sure your `labels.txt` file format is correct. It should be:
```
0 Stop
1 Be Ready
2 Go
```

### 7. Run the Application
```bash
python app.py
```

The application will start and you can access it at: `http://localhost:5000`

## 🎯 Features

### Real-time Detection
- Live camera feed with OpenCV
- Real-time traffic light classification
- Dynamic background colors based on detection

### Interactive Dashboard
- Modern, responsive UI with Bootstrap
- Animated status cards
- Confidence score visualization
- Detection statistics counter
- Start/Stop detection controls

### Visual Feedback
- **Red Light (Stop)**: Red background with stop icon and pulsing animation
- **Yellow Light (Be Ready)**: Yellow background with warning icon
- **Green Light (Go)**: Blue-green gradient with arrow icon
- Confidence bar showing prediction accuracy

### Control Features
- Toggle detection on/off
- Real-time status updates
- Detection statistics tracking
- Smooth animations and transitions

## 🔧 Troubleshooting

### Camera Issues
- Make sure your camera is not being used by another application
- Try changing the camera index in `cv2.VideoCapture(0)` to `cv2.VideoCapture(1)` if you have multiple cameras

### Model Loading Issues
- Ensure `keras_Model.h5` and `labels.txt` are in the root directory
- Check that the model file is not corrupted
- Verify TensorFlow/Keras versions compatibility

### Performance Issues
- Reduce camera resolution if needed
- Adjust prediction interval in the JavaScript code
- Consider running on a machine with better GPU support

## 🎨 Customization

### Changing Colors
Edit the CSS variables in the `<style>` section of `dashboard.html`:
- `.status-card.stop` - Red light styling
- `.status-card.ready` - Yellow light styling  
- `.status-card.go` - Green light styling

### Adjusting Detection Sensitivity
Modify the confidence threshold in `app.py` if needed:
```python
if confidence_score > 0.7:  # Adjust threshold as needed
    # Process prediction
```

### Adding More Classes
If you retrain your model with more classes:
1. Update `labels.txt`
2. Add corresponding color mappings in `app.py`
3. Add new CSS classes in `dashboard.html`

## 📱 Browser Compatibility
- Chrome (Recommended)
- Firefox
- Safari
- Edge

## 🚀 Deployment
For production deployment, consider using:
- Gunicorn + Nginx
- Docker containers
- Cloud platforms (Heroku, AWS, etc.)

---

**Note**: Make sure your camera permissions are enabled for the browser and that your model files are properly placed in the project directory before running the application.