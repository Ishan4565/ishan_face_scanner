import streamlit as st
from streamlit_webrtc import webrtc_streamer
import cv2
import numpy as np
import os
from datetime import datetime
import mediapipe as mp
import pickle

DATASET_PATH = os.path.join("dataset", "Ishan")
st.title("Ishan Face Scanner 🚀")

# --- Load MediaPipe ---
mp_face_detection = mp.solutions.face_detection
mp_face_mesh = mp.solutions.face_mesh

@st.cache_resource
def load_known_embeddings():
    """Build simple color histogram embeddings from dataset images."""
    embeddings = []
    names = []
    if not os.path.exists(DATASET_PATH):
        return embeddings, names

    with mp_face_detection.FaceDetection(
        model_selection=0, min_detection_confidence=0.5
    ) as detector:
        for filename in os.listdir(DATASET_PATH):
            if not filename.lower().endswith((".jpg", ".png", ".jpeg")):
                continue
            path = os.path.join(DATASET_PATH, filename)
            img = cv2.imread(path)
            if img is None:
                continue
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = detector.process(rgb)
            if not results.detections:
                continue
            # Crop face region
            d = results.detections[0]
            bb = d.location_data.relative_bounding_box
            h, w = img.shape[:2]
            x1 = max(0, int(bb.xmin * w))
            y1 = max(0, int(bb.ymin * h))
            x2 = min(w, int((bb.xmin + bb.width) * w))
            y2 = min(h, int((bb.ymin + bb.height) * h))
            face_crop = rgb[y1:y2, x1:x2]
            if face_crop.size == 0:
                continue
            # Build embedding as flattened resized face
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
        self.detector = mp_face_detection.FaceDetection(
            model_selection=0, min_detection_confidence=0.5
        )

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        self.frame_count += 1

        if self.frame_count % 8 == 0:
            small = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
            rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
            results = self.detector.process(rgb)

            self.last_boxes = []
            self.last_names = []

            if results.detections:
                h, w = small.shape[:2]
                for detection in results.detections:
                    bb = detection.location_data.relative_bounding_box
                    x1 = max(0, int(bb.xmin * w))
                    y1 = max(0, int(bb.ymin * h))
                    x2 = min(w, int((bb.xmin + bb.width) * w))
                    y2 = min(h, int((bb.ymin + bb.height) * h))

                    face_crop = rgb[y1:y2, x1:x2]
                    name = "Unknown"

                    if face_crop.size > 0 and known_embeddings:
                        face_resized = cv2.resize(face_crop, (64, 64)).flatten().astype(np.float32)
                        face_norm = face_resized / (np.linalg.norm(face_resized) + 1e-6)
                        sims = [np.dot(face_norm, ke) for ke in known_embeddings]
                        best_idx = int(np.argmax(sims))
                        if sims[best_idx] > 0.75:
                            name = known_names[best_idx]

                    # Scale boxes back up (we resized by 0.5)
                    self.last_boxes.append((x1*2, y1*2, x2*2, y2*2))
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
