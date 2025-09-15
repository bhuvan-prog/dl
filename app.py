from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import pickle
from PIL import Image
import io
import base64
import re
import os

app = Flask(__name__)

# Global variable for the model
model = None

# Load the trained model
def load_model():
    global model  # Declare we're using the global model variable
    try:
        with open('mask_detector.pkl', 'rb') as f:
            model = pickle.load(f)
        print("Model loaded successfully from pickle file")
        return model
    except:
        try:
            import tensorflow as tf
            model = tf.keras.models.load_model("mask_detector.h5")
            print("Model loaded successfully from h5 file")
            return model
        except Exception as e:
            print(f"Error: Could not load model. Please train the model first. Error: {e}")
            return None

# Load model when the app starts
load_model()

# Load face cascade classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

labels_dict = {0: 'No Mask', 1: 'Mask'}
color_dict = {0: (0, 0, 255), 1: (0, 255, 0)}

# Global variables for statistics
mask_count = 0
no_mask_count = 0

def detect_mask(image_data):
    global mask_count, no_mask_count
    
    if model is None:
        return [], None, "Model not loaded"
    
    # Convert base64 image data to OpenCV format
    image_data = re.sub('^data:image/.+;base64,', '', image_data)
    image_bytes = io.BytesIO(base64.b64decode(image_data))
    image = Image.open(image_bytes)
    frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    results = []
    
    for (x, y, w, h) in faces:
        face_img = frame[y:y+h, x:x+w]
        resized = cv2.resize(face_img, (120, 120))
        resized = resized / 255.0
        reshaped = np.reshape(resized, (1, 120, 120, 3))
        
        # Use the model to predict
        result = model.predict(reshaped)
        label = 1 if result[0][0] > 0.5 else 0
        confidence = float(result[0][0] if label == 1 else 1 - result[0][0])
        
        # Update counts
        if label == 1:
            mask_count += 1
        else:
            no_mask_count += 1
        
        results.append({
            'box': [int(x), int(y), int(w), int(h)],
            'label': int(label),
            'confidence': confidence,
            'label_text': labels_dict[label]
        })
        
        # Draw rectangle and label on the image
        cv2.rectangle(frame, (x, y), (x+w, y+h), color_dict[label], 2)
        cv2.putText(frame, labels_dict[label], (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color_dict[label], 2)
    
    # Convert the processed image back to base64
    _, buffer = cv2.imencode('.jpg', frame)
    processed_image = base64.b64encode(buffer).decode('utf-8')
    
    return results, processed_image, None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({'success': False, 'error': 'No image data provided'})
        
        image_data = data['image']
        
        # Detect masks in the image
        results, processed_image, error = detect_mask(image_data)
        
        if error:
            return jsonify({'success': False, 'error': error})
        
        total = mask_count + no_mask_count
        accuracy = (mask_count / total * 100) if total > 0 else 0
        
        return jsonify({
            'success': True,
            'results': results,
            'processed_image': f"data:image/jpeg;base64,{processed_image}",
            'stats': {
                'mask_count': mask_count,
                'no_mask_count': no_mask_count,
                'accuracy': round(accuracy, 2)
            }
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/stats')
def get_stats():
    total = mask_count + no_mask_count
    accuracy = (mask_count / total * 100) if total > 0 else 0
    
    return jsonify({
        'mask_count': mask_count,
        'no_mask_count': no_mask_count,
        'accuracy': round(accuracy, 2)
    })

@app.route('/train', methods=['POST'])
def train_model():
    global model  # Declare we're using the global model variable
    try:
        # Import and run the training script
        from train_model import build_and_train_model
        new_model, accuracy = build_and_train_model()
        
        # Update the global model reference
        model = new_model
        
        return jsonify({
            'success': True,
            'accuracy': float(accuracy)
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)