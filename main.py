import cv2
import numpy as np
import pyttsx3
import tkinter as tk
from tkinter import StringVar, Label, Button, Frame
from PIL import Image, ImageTk
import threading
import time
from tensorflow.keras.models import load_model

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
stabilization_buffer = []
stable_char = ""
word_buffer = ""
sentence = ""

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
    global word_buffer, sentence
    word_buffer = ""
    sentence = ""
    current_word.set("N/A")
    current_sentence.set("N/A")
    current_alphabet.set("N/A")

def toggle_pause():
    global is_paused
    is_paused = not is_paused
    pause_button.config(text="Resume" if is_paused else "Pause")

def preprocess_frame(frame):
    """Preprocess frame for CNN model"""
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame, (64, 64))
    frame = frame / 255.0
    return np.expand_dims(frame, axis=0)

# Video Capture
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 400)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 300)

# Variables for stabilization timing
last_registered_time = time.time()
registration_delay = 1.5  # Minimum delay before registering same character

def process_frame():
    global stabilization_buffer, stable_char, word_buffer, sentence, last_registered_time

    ret, frame = cap.read()
    if not ret:
        root.after(10, process_frame)
        return

    frame = cv2.flip(frame, 1)  # Mirror view
    
    if is_paused:
        # Show paused state
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
        
        # Display prediction info on frame
        cv2.putText(frame, f"Letter: {stable_char}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.putText(frame, "Show ASL sign to camera", (10, frame.shape[0]-20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Update video feed in GUI
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
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
    root.destroy()

root.protocol("WM_DELETE_WINDOW", on_closing)
root.mainloop()