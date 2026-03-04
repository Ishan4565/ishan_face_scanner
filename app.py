import streamlit as st
from streamlit_webrtc import webrtc_streamer
import cv2
import numpy as np
import os
import pickle
from datetime import datetime
import insightface
from insightface.app import FaceAnalysis

DATASET_PATH = os.path.join("dataset", "Ishan")
EMBEDDINGS_FILE = "embeddings.pkl"

st.title("Ishan Face Scanner 🚀")

@st.cache_resource
def load_model():
    app = FaceAnalysis(name="buffalo_sc", providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=0, det_size=(320, 320))
    return app

@st.cache_resource
def get_known_embeddings(_app):
    embeddings = []
    names = []
    if os.path.exists(DATASET_PATH):
        for filename in os.listdir(DATASET_PATH):
            if filename.lower().endswith((".jpg", ".png", ".jpeg")):
                img = cv2.imread(os.path.join(DATASET_PATH, filename))
                if img is None:
                    continue
                faces = _app.get(img)
                if faces:
                    embeddings.append(faces[0].normed_embedding)
                    names.append("Ishan")
    return embeddings, names

face_app = load_model()
known_embeddings, known_names = get_known_embeddings(face_app)

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
            faces = face_app.get(small)

            self.last_boxes = []
            self.last_names = []

            for face in faces:
                embedding = face.normed_embedding
                name = "Unknown"

                if known_embeddings:
                    sims = [np.dot(embedding, ke) for ke in known_embeddings]
                    best_idx = int(np.argmax(sims))
                    if sims[best_idx] > 0.35:
                        name = known_names[best_idx]

                box = face.bbox.astype(int) * 2  # scale back up
                self.last_boxes.append(box)
                self.last_names.append(name)

        now = datetime.now().strftime("%H:%M:%S")
        for box, name in zip(self.last_boxes, self.last_names):
            x1, y1, x2, y2 = box
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
