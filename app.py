import streamlit as st
from streamlit_webrtc import webrtc_streamer
import cv2
import numpy as np
import os
from datetime import datetime

DATASET_PATH = os.path.join("dataset", "Ishan")
st.title("Ishan Face Scanner 🚀")

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

@st.cache_resource
def load_known_embeddings():
    embeddings = []
    names = []
    if not os.path.exists(DATASET_PATH):
        st.warning("Dataset folder not found!")
        return embeddings, names
    for filename in os.listdir(DATASET_PATH):
        if not filename.lower().endswith((".jpg", ".png", ".jpeg")):
            continue
        img = cv2.imread(os.path.join(DATASET_PATH, filename))
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        if len(faces) == 0:
            continue
        x, y, w, h = faces[0]
        face_crop = img[y:y+h, x:x+w]
        face_resized = cv2.resize(face_crop, (64, 64)).flatten().astype(np.float32)
        face_norm = face_resized / (np.linalg.norm(face_resized) + 1e-6)
        embeddings.append(face_norm)
        names.append("Ishan")
    return embeddings, names

known_embeddings, known_names = load_known_embeddings()

class VideoProcessor:
    def __init__(self):
        self.frame_count = 0
        self.last_boxes = []
        self.last_names = []

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        self.frame_count += 1

        if self.frame_count % 8 == 0:
            small = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 4, minSize=(30, 30))
            self.last_boxes = []
            self.last_names = []

            for (x, y, w, h) in faces:
                face_crop = small[y:y+h, x:x+w]
                name = "Unknown"
                if face_crop.size > 0 and known_embeddings:
                    face_resized = cv2.resize(face_crop, (64, 64)).flatten().astype(np.float32)
                    face_norm = face_resized / (np.linalg.norm(face_resized) + 1e-6)
                    sims = [np.dot(face_norm, ke) for ke in known_embeddings]
                    best_idx = int(np.argmax(sims))
                    if sims[best_idx] > 0.75:
                        name = known_names[best_idx]
                self.last_boxes.append((x*2, y*2, (x+w)*2, (y+h)*2))
                self.last_names.append(name)

        now = datetime.now().strftime("%H:%M:%S")
        for (x1, y1, x2, y2), name in zip(self.last_boxes, self.last_names):
            color = (0, 255, 0) if name == "Ishan" else (0, 0, 255)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img, f"{name} | {now}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        return frame.from_ndarray(img, format="bgr24")

webrtc_streamer(
    key="face-scanner",
    video_processor_factory=VideoProcessor,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)
```

Also update `requirements.txt` — remove mediapipe completely since we no longer need it:
```
streamlit
streamlit-webrtc
opencv-python-headless
numpy
av
