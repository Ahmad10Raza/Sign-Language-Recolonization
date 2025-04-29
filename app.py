from flask import Flask, request, jsonify
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Load model
model = load_model('asl_model.h5')

# Complete list of ASL classes (29 classes)
class_names = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 
    'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
    'W', 'X', 'Y', 'Z', 'del', 'nothing', 'space'
]

# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/predict', methods=['POST'])
def predict():
    # Check if file was uploaded
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    
    # Check for empty filename
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    # Check file extension
    if not allowed_file(file.filename):
        return jsonify({
            'error': 'Invalid file type',
            'allowed_extensions': list(ALLOWED_EXTENSIONS)
        }), 400
    
    try:
        # Read and preprocess image
        img_bytes = file.read()
        img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
        
        if img is None:
            return jsonify({'error': 'Invalid image file'}), 400
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (64, 64))
        img = img / 255.0
        img = np.expand_dims(img, axis=0)
        
        # Make prediction
        pred = model.predict(img)
        predicted_class = class_names[np.argmax(pred)]
        confidence = float(np.max(pred))
        
        return jsonify({
            'prediction': predicted_class,
            'confidence': confidence,
            'all_predictions': {class_names[i]: float(pred[0][i]) for i in range(len(class_names))}
        })
    
    except Exception as e:
        return jsonify({
            'error': 'Prediction failed',
            'message': str(e)
        }), 500

@app.route('/')
def home():
    return jsonify({
        'message': 'ASL Recognition API',
        'endpoints': {
            '/predict': 'POST image file for ASL prediction'
        },
        'model_info': {
            'input_shape': [64, 64, 3],
            'classes': class_names
        }
    })

if __name__ == '__main__':
    # Create uploads directory if not exists
    os.makedirs('uploads', exist_ok=True)
    
    # Run with production settings
    app.run(host='0.0.0.0', port=5000, debug=False)