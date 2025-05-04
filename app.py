import cv2
import numpy as np
import pyttsx3
import tkinter as tk
from tkinter import StringVar, Label, Button, Frame
from PIL import Image, ImageTk
import threading
import time
import mediapipe as mp
from mediapipe.python.solutions import drawing_utils as mp_drawing
from mediapipe.python.solutions.hands_connections import HAND_CONNECTIONS
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Initialize MediaPipe Hands with drawing utilities
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.5
)

# Custom drawing styles
hand_landmarks_style = mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2)
hand_connections_style = mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2)

# Load your trained CNN model
model = load_model('best_enhanced_model.h5')
class_names = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 
    'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
    'W', 'X', 'Y', 'Z', 'del', 'nothing', 'space'
]

# Text-to-Speech setup
engine = pyttsx3.init()

# Initialize buffers and history
prediction_history = []
stable_char = ""
word_buffer = ""
sentence = ""
history_length = 15  # Number of frames to average
min_confidence = 0.60  # Only accept predictions with 85%+ confidence

# GUI Setup
root = tk.Tk()
root.title("ASL Recognition with CNN")
root.geometry("1300x650")
root.configure(bg="#2c2f33")  # Dark theme
root.resizable(True, True)

# Variables for GUI
current_alphabet = StringVar(value="N/A")
current_word = StringVar(value="N/A")
current_sentence = StringVar(value="N/A")
is_paused = False

# Title
title_label = Label(root, text="ASL Recognition with CNN", font=("Arial", 28, "bold"), 
                   fg="#ffffff", bg="#2c2f33")
title_label.grid(row=0, column=0, columnspan=2, pady=10)

# Layout Frames
video_frame = Frame(root, bg="#2c2f33", bd=5, relief="solid", width=500, height=400)
video_frame.grid(row=1, column=0, rowspan=3, padx=20, pady=20)
video_frame.grid_propagate(False)

content_frame = Frame(root, bg="#2c2f33")
content_frame.grid(row=1, column=1, sticky="n", padx=(20, 40), pady=(60, 20))

button_frame = Frame(root, bg="#2c2f33")
button_frame.grid(row=3, column=1, pady=(10, 20), padx=(10, 20), sticky="n")

# Video feed
video_label = tk.Label(video_frame)
video_label.pack(expand=True)

# Labels
Label(content_frame, text="Current Letter:", font=("Arial", 20), 
      fg="#ffffff", bg="#2c2f33").pack(anchor="w", pady=(0, 10))
Label(content_frame, textvariable=current_alphabet, font=("Arial", 24, "bold"), 
      fg="#1abc9c", bg="#2c2f33").pack(anchor="center")

Label(content_frame, text="Current Word:", font=("Arial", 20), 
      fg="#ffffff", bg="#2c2f33").pack(anchor="w", pady=(20, 10))
Label(content_frame, textvariable=current_word, font=("Arial", 20), 
      fg="#f39c12", bg="#2c2f33", wraplength=500, justify="left").pack(anchor="center")

Label(content_frame, text="Current Sentence:", font=("Arial", 20), 
      fg="#ffffff", bg="#2c2f33").pack(anchor="w", pady=(20, 10))
Label(content_frame, textvariable=current_sentence, font=("Arial", 20), 
      fg="#9b59b6", bg="#2c2f33", wraplength=500, justify="left").pack(anchor="center")

def speak_text(text):
    def tts_thread():
        engine.say(text)
        engine.runAndWait()
    threading.Thread(target=tts_thread, daemon=True).start()

def reset_sentence():
    global word_buffer, sentence, prediction_history
    word_buffer = ""
    sentence = ""
    prediction_history = []
    current_word.set("N/A")
    current_sentence.set("N/A")
    current_alphabet.set("N/A")

def toggle_pause():
    global is_paused
    is_paused = not is_paused
    pause_button.config(text="Resume" if is_paused else "Pause")

def detect_and_crop_hand(frame):
    """Detect hand using MediaPipe and return cropped hand region"""
    results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    hand_landmarks = None
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Get hand bounding box coordinates
            h, w = frame.shape[:2]
            x_coords = [lm.x * w for lm in hand_landmarks.landmark]
            y_coords = [lm.y * h for lm in hand_landmarks.landmark]
            x_min, x_max = int(min(x_coords)), int(max(x_coords))
            y_min, y_max = int(min(y_coords)), int(max(y_coords))
            
            # Add 20% padding
            padding = int(0.2 * max(x_max - x_min, y_max - y_min))
            x_min = max(0, x_min - padding)
            x_max = min(w, x_max + padding)
            y_min = max(0, y_min - padding)
            y_max = min(h, y_max + padding)
            
            # Crop hand region
            hand_roi = frame[y_min:y_max, x_min:x_max]
            
            return hand_roi, hand_landmarks, (x_min, y_min, x_max, y_max)
    
    return None, None, None

def preprocess_frame(frame):
    """Preprocess frame for CNN model"""
    # Convert to HSV and normalize brightness
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    frame[:,:,2] = cv2.equalizeHist(frame[:,:,2])
    frame = cv2.cvtColor(frame, cv2.COLOR_HSV2RGB)
    
    # Resize and normalize
    frame = cv2.resize(frame, (64, 64))
    frame = frame / 255.0
    
    # Apply slight augmentation (like during training)
    aug = ImageDataGenerator(
        rotation_range=8,
        width_shift_range=0.05,
        height_shift_range=0.05,
        zoom_range=0.05
    )
    frame = aug.random_transform(frame)
    
    return np.expand_dims(frame, axis=0)

def get_stable_prediction(current_pred, confidence):
    """Apply temporal smoothing to predictions"""
    global prediction_history
    
    if confidence < min_confidence:
        return None
    
    prediction_history.append(current_pred)
    if len(prediction_history) > history_length:
        prediction_history.pop(0)
    
    # Only return prediction if it appears consistently
    if prediction_history.count(current_pred) >= history_length//2:
        return current_pred
    return None

# Video Capture
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

def process_frame():
    global stable_char, word_buffer, sentence
    
    ret, frame = cap.read()
    if not ret:
        root.after(10, process_frame)
        return

    frame = cv2.flip(frame, 1)  # Mirror view
    display_frame = frame.copy()
    
    if is_paused:
        cv2.putText(display_frame, "PAUSED", (50, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    else:
        # Detect and crop hand
        hand_roi, hand_landmarks, bbox = detect_and_crop_hand(frame)
        
        if hand_roi is not None:
            # Draw hand landmarks with custom styles
            mp_drawing.draw_landmarks(
                display_frame,
                hand_landmarks,
                HAND_CONNECTIONS,
                hand_landmarks_style,
                hand_connections_style
            )
            
            # Draw bounding box
            x_min, y_min, x_max, y_max = bbox
            cv2.rectangle(display_frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            
            # Preprocess and predict
            processed = preprocess_frame(hand_roi)
            predictions = model.predict(processed, verbose=0)
            confidence = np.max(predictions)
            predicted_class = class_names[np.argmax(predictions)]
            
            # Stabilize prediction
            stable_pred = get_stable_prediction(predicted_class, confidence)
            
            if stable_pred is not None:
                stable_char = stable_pred
                current_alphabet.set(stable_char)
                
                # Handle word/sentence formation
                if stable_char == 'space':
                    if word_buffer.strip():
                        speak_text(word_buffer)
                        sentence += word_buffer + " "
                        current_sentence.set(sentence.strip())
                    word_buffer = ""
                    current_word.set("N/A")
                elif stable_char == 'del':
                    word_buffer = word_buffer[:-1] if word_buffer else ""
                    current_word.set(word_buffer if word_buffer else "N/A")
                elif stable_char != 'nothing':
                    word_buffer += stable_char
                    current_word.set(word_buffer)
    
    # Display prediction info
    cv2.putText(display_frame, f"Letter: {stable_char}", (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    cv2.putText(display_frame, "Show your hand to the camera", (10, display_frame.shape[0]-20),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Update video feed
    img = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    img_tk = ImageTk.PhotoImage(image=img)
    video_label.imgtk = img_tk
    video_label.configure(image=img_tk)

    root.after(10, process_frame)

# Buttons
Button(button_frame, text="Reset", font=("Arial", 16), command=reset_sentence, 
      bg="#e74c3c", fg="#ffffff", relief="flat", height=2, width=14).grid(row=0, column=0, padx=10)

pause_button = Button(button_frame, text="Pause", font=("Arial", 16), command=toggle_pause, 
                     bg="#3498db", fg="#ffffff", relief="flat", height=2, width=12)
pause_button.grid(row=0, column=1, padx=10)

Button(button_frame, text="Speak Word", font=("Arial", 16), 
      command=lambda: speak_text(current_word.get()), 
      bg="#27ae60", fg="#ffffff", relief="flat", height=2, width=14).grid(row=0, column=2, padx=10)

Button(button_frame, text="Speak Sentence", font=("Arial", 16), 
      command=lambda: speak_text(current_sentence.get()), 
      bg="#9b59b6", fg="#ffffff", relief="flat", height=2, width=14).grid(row=0, column=3, padx=10)

# Start processing frames
process_frame()

# Cleanup on exit
def on_closing():
    cap.release()
    cv2.destroyAllWindows()
    root.destroy()

root.protocol("WM_DELETE_WINDOW", on_closing)
root.mainloop()