import streamlit as st
from streamlit_webrtc import webrtc_streamer
import cv2
import numpy as np
import os
from deepface import DeepFace
from datetime import datetime
import av

DATASET_PATH = os.path.join("dataset", "Ishan")
DB_PATH = "dataset"  # DeepFace searches recursively

st.title("Ishan Face Scanner 🚀")

class VideoProcessor:
    def __init__(self):
        self.frame_count = 0
        self.last_locations = []
        self.last_names = []

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        self.frame_count += 1

        if self.frame_count % 10 == 0:  # process every 10th frame
            small = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)

            try:
                results = DeepFace.find(
                    img_path=small,
                    db_path=DB_PATH,
                    model_name="Facenet",       # lightweight & accurate
                    detector_backend="opencv",  # fastest detector
                    enforce_detection=False,
                    silent=True
                )

                self.last_locations = []
                self.last_names = []

                for result_df in results:
                    if not result_df.empty:
                        # Get face region from DeepFace result
                        row = result_df.iloc[0]
                        identity = row["identity"]

                        # Extract name from path
                        name = os.path.basename(os.path.dirname(identity))

                        # Get facial area if available
                        if "source_x" in row:
                            x = int(row["source_x"] * 2)  # scale back up
                            y = int(row["source_y"] * 2)
                            w = int(row["source_w"] * 2)
                            h = int(row["source_h"] * 2)
                            self.last_locations.append((x, y, w, h))
                        else:
                            self.last_locations.append(None)

                        self.last_names.append(name)

            except Exception:
                # No face detected or DB error — just skip
                pass

        # Draw results on every frame
        now = datetime.now().strftime("%H:%M:%S")
        for loc, name in zip(self.last_locations, self.last_names):
            if loc is None:
                continue
            x, y, w, h = loc
            color = (0, 255, 0) if name == "Ishan" else (0, 0, 255)
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(img, f"{name} | {now}", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        return frame.from_ndarray(img, format="bgr24")


webrtc_streamer(
    key="face-scanner",
    video_processor_factory=VideoProcessor,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)
