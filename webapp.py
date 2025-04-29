import time
import threading
import cv2
import numpy as np
import pyttsx3
from flask import Flask, render_template, Response, jsonify, request
from tensorflow.keras.models import load_model

# Initialize Flask app
app = Flask(__name__)

# ---------------------------
# Load your trained CNN model
# ---------------------------
model = load_model('asl_model.h5')  # Your CNN model
class_names = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 
    'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
    'W', 'X', 'Y', 'Z', 'del', 'nothing', 'space'
]

# ---------------------------
# Text-to-Speech Setup
# ---------------------------
engine = pyttsx3.init()

def speak_text(text):
    """Speak text in a separate thread"""
    def tts_thread():
        engine.say(text)
        engine.runAndWait()
    threading.Thread(target=tts_thread, daemon=True).start()

# ---------------------------
# Global Variables
# ---------------------------
stabilization_buffer = []
stable_char = ""
word_buffer = ""
sentence = ""
last_registered_time = time.time()
registration_delay = 1.5
is_paused = False

# ---------------------------
# Webcam Setup
# ---------------------------
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

def preprocess_frame(frame):
    """Preprocess frame for your CNN model"""
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame, (64, 64))
    frame = frame / 255.0
    return np.expand_dims(frame, axis=0)

def gen_frames():
    """Video streaming generator function"""
    global stabilization_buffer, stable_char, word_buffer, sentence, last_registered_time, is_paused
    
    while True:
        success, frame = cap.read()
        if not success:
            break
            
        frame = cv2.flip(frame, 1)  # Mirror view
        
        if is_paused:
            cv2.putText(frame, "PAUSED", (50, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            # Preprocess and predict
            processed = preprocess_frame(frame)
            predictions = model.predict(processed)
            predicted_class = class_names[np.argmax(predictions)]
            confidence = np.max(predictions)
            
            # Only consider predictions with high confidence
            if confidence > 0.8:
                stabilization_buffer.append(predicted_class)
                if len(stabilization_buffer) > 30:
                    stabilization_buffer.pop(0)
                
                # Get most frequent prediction in buffer
                current_char = max(set(stabilization_buffer), 
                                  key=stabilization_buffer.count)
                
                if stabilization_buffer.count(current_char) > 25:
                    current_time = time.time()
                    if current_time - last_registered_time > registration_delay:
                        stable_char = current_char
                        last_registered_time = current_time
                        
                        # Handle word/sentence formation
                        if stable_char == 'space':
                            if word_buffer.strip():
                                speak_text(word_buffer)
                                sentence += word_buffer + " "
                            word_buffer = ""
                        elif stable_char == 'del':
                            word_buffer = word_buffer[:-1] if word_buffer else ""
                        elif stable_char != 'nothing':
                            word_buffer += stable_char
                        
            # Display information
            cv2.putText(frame, f"Letter: {stable_char}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            cv2.putText(frame, f"Word: {word_buffer}", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(frame, f"Sentence: {sentence}", (10, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
            cv2.putText(frame, "Show your hand to the camera", (10, 450),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Encode frame
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

# ---------------------------
# Flask Routes
# ---------------------------
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/toggle_pause', methods=['POST'])
def toggle_pause():
    global is_paused
    is_paused = not is_paused
    return jsonify({'paused': is_paused})

@app.route('/reset', methods=['POST'])
def reset():
    global word_buffer, sentence
    word_buffer = ""
    sentence = ""
    return jsonify({'word': word_buffer, 'sentence': sentence})

@app.route('/get_text')
def get_text():
    return jsonify({
        'letter': stable_char,
        'word': word_buffer,
        'sentence': sentence
    })

@app.route('/speak_sentence', methods=['POST'])
def speak_sentence():
    speak_text(sentence)
    return jsonify({'sentence': sentence})

@app.route('/speak_word', methods=['POST'])
def speak_word():
    speak_text(word_buffer)
    return jsonify({'word': word_buffer})

# ---------------------------
# Cleanup on exit
# ---------------------------
def cleanup():
    cap.release()
    cv2.destroyAllWindows()

import atexit
atexit.register(cleanup)

if __name__ == '__main__':
    app.run(debug=True)