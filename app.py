import streamlit as st
from streamlit_webrtc import webrtc_streamer
import face_recognition
import cv2
import numpy as np
import os
from datetime import datetime

# --- LOAD DATABASE ONCE ---
base_path = os.path.join("dataset", "Ishan")

@st.cache_resource
def get_known_data():
    encodings = []
    if os.path.exists(base_path):
        for filename in os.listdir(base_path):
            if filename.lower().endswith((".jpg", ".png", ".jpeg")):
                img = face_recognition.load_image_file(os.path.join(base_path, filename))
                encs = face_recognition.face_encodings(img)
                if encs:
                    encodings.append(encs[0])
    return encodings

known_encs = get_known_data()

class VideoProcessor:
    def __init__(self):
        self.frame_count = 0
        self.last_locations = []
        self.last_names = []

    def recv(self, frame):
    img = frame.to_ndarray(format="bgr24")
    
    if not known_encs:  # ← ADD THIS GUARD
        return frame.from_ndarray(img, format="bgr24")

        # ONLY process every 5th frame to save 80% CPU
        if self.frame_count % 5 == 0:
            # 1. Shrink even more (0.2 instead of 0.25)
            small = cv2.resize(img, (0, 0), fx=0.2, fy=0.2)
            rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

            # 2. Use the fastest model ('hog')
            self.last_locations = face_recognition.face_locations(rgb_small, model="hog")
            face_encodings = face_recognition.face_encodings(rgb_small, self.last_locations)

            self.last_names = []
            for face_encoding in face_encodings:
                matches = face_recognition.compare_faces(known_encs, face_encoding, 0.6)
                name = "Ishan" if True in matches else "Unknown"
                self.last_names.append(name)

        # 3. Draw the last known positions on EVERY frame (smooth visuals)
        now = datetime.now().strftime("%H:%M:%S")
        for (top, right, bottom, left), name in zip(self.last_locations, self.last_names):
            # Scale back up (we resized by 0.2, so multiply by 5)
            top, right, bottom, left = top*5, right*5, bottom*5, left*5
            
            color = (0, 255, 0) if name == "Ishan" else (0, 0, 255)
            cv2.rectangle(img, (left, top), (right, bottom), color, 2)
            cv2.putText(img, f"{name} | {now}", (left, top - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        return frame.from_ndarray(img, format="bgr24")


webrtc_streamer(key="fast-rec", video_processor_factory=VideoProcessor)
