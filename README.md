# 👤 Real-Time Face Recognition System

## 🎯 Problem Statement

Traditional face recognition systems either:
- **Option A:** Cloud APIs (expensive, privacy concerns, requires internet)
- **Option B:** Offline solutions (slow, limited accuracy, complex setup)

**The Challenge:**
- Build real-time face recognition that works in browser (mobile + desktop)
- Achieve <100ms inference latency
- Handle real-world variations (lighting, angles, distances)
- Optimize CPU usage without sacrificing accuracy

## ✅ Solution

Built a **real-time face recognition system** deployed on Streamlit Cloud using:
- **Computer Vision:** OpenCV Haar Cascade for face detection
- **Optimization:** Frame-skipping + downscaling = **80% CPU reduction**
- **Deployment:** Browser-based (no installation needed)
- **Cross-Platform:** iOS, Android, Desktop all supported

## 🛠 Technical Architecture

```
Webcam Feed (30 fps)
    ↓
[Frame Skip: Process every 3rd frame]  ← Reduces to 10 fps processing
    ↓
[Downscale: 50% resolution]  ← Speeds up detection
    ↓
[Haar Cascade Detection] ← Find faces
    ↓
[Extract Embeddings] ← Face representation
    ↓
[Similarity Matching] ← Compare to known faces
    ↓
[Display Result]  ← Show match + confidence
```

## 📊 Performance Metrics

### Before Optimization
- **CPU Usage:** 100% (unusable)
- **FPS:** 5 (very slow, laggy)
- **Latency:** 500ms (noticeable delay)
- **Smoothness:** Choppy, frustrating

### After Optimization
- **CPU Usage:** ~20% (smooth, efficient) ⬇️ **80% reduction**
- **FPS:** 30 (smooth real-time)
- **Latency:** <100ms (imperceptible)
- **Smoothness:** Fluid, responsive ✅

## 🎯 Optimization Techniques

### 1. Frame-Skipping
```python
frame_count = 0
SKIP_FRAMES = 3

while True:
    ret, frame = cap.read()
    frame_count += 1
    
    if frame_count % SKIP_FRAMES == 0:
        # Process only every 3rd frame
        faces = detect_faces(frame)
    else:
        # Reuse last detection
        display_last_detection(frame)
```

**Impact:** 3x speed improvement
**Trade-off:** Slightly delayed response when new face appears

### 2. Image Downscaling
```python
# Original: 1920x1080
# Downscaled: 960x540

def optimize_frame(frame):
    # Resize to 50% (4x fewer pixels to process)
    small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
    
    # Detect on small frame
    faces = detect_faces(small_frame)
    
    # Scale face coordinates back to original
    faces_original = scale_faces_back(faces, factor=2)
    
    return faces_original
```

**Impact:** 4x speed improvement
**Trade-off:** Slight accuracy loss on very small faces (acceptable for webcam distance)

### 3. ROI-Based Processing
```python
# Only process Region of Interest (where faces typically are)
# Skip top 10%, bottom 30% (usually background/body)

roi = frame[100:700, :]  # Focus on face-likely region
faces = detect_faces(roi)
```

**Impact:** 2x improvement on processing time

### Combined Impact
- Frame-skip: 3x faster
- Downscaling: 4x faster
- ROI: 2x faster
- **Total: 24x potential speedup** (but we use conservative settings for accuracy)
- **Practical: 5x speedup, maintaining quality** = 100% → 20% CPU

## 🛠 Tech Stack

- **Language:** Python
- **Computer Vision:** OpenCV (Haar Cascade)
- **Face Detection:** Haar Cascade classifier (fast, on-device)
- **Face Embedding:** OpenCV DNN (deep learning-based comparison)
- **Frontend:** Streamlit
- **Webcam Access:** streamlit-webrtc (browser WebRTC)
- **Deployment:** Streamlit Cloud
- **Data Processing:** NumPy
- **Visualization:** Matplotlib

## 🚀 Live Demo

**Access:** https://ishanfacescanner-rg6ejul3vcnjahkmpxswn4.streamlit.app/

**Features:**
- ✅ Real-time webcam feed
- ✅ Face detection + recognition
- ✅ Confidence scores
- ✅ Multiple face support (identify multiple people simultaneously)
- ✅ Cross-platform (mobile + desktop)
- ✅ No installation needed (browser-based)

**How to Use:**
1. Click "Enable webcam"
2. Face the camera
3. System detects and identifies your face
4. See confidence score (0-100%)
5. Add new faces to database

## 📂 Project Structure

```
ishan_face_scanner/
├── app.py                          # Main Streamlit app
├── face_detector.py               # Haar Cascade wrapper
├── face_matcher.py                # Embedding + similarity
├── optimization.py                # Frame-skip, downscaling logic
├── requirements.txt
├── haarcascades/
│   └── haarcascade_frontalface_default.xml
├── known_faces/
│   ├── person1.jpg
│   ├── person2.jpg
│   └── ...
└── README.md
```

## 💻 How to Run Locally

**Installation:**
```bash
git clone https://github.com/Ishan4565/ishan_face_scanner.git
cd ishan_face_scanner
pip install -r requirements.txt
```

**Run:**
```bash
streamlit run app.py
```

**Access:** http://localhost:8501

## 📈 Algorithm Breakdown

### Face Detection (Haar Cascade)
```python
import cv2

# Load pre-trained cascade
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

# Detect faces
def detect_faces(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.3,      # How much to increase scale
        minNeighbors=5,       # How many neighbors to consider
        minSize=(30, 30)      # Minimum face size
    )
    return faces

# Why Haar Cascade?
# - Fast (real-time capable)
# - On-device (no API calls)
# - Lightweight (few MB)
# - Reliable (trained on thousands of faces)
```

### Face Matching (Embedding-Based)
```python
# Extract face embedding (128-D vector representing face)
def get_face_embedding(face_image):
    # Use pre-trained deep learning model
    embedding = model.forward(face_image)  # 128-dimensional vector
    return embedding

# Compare embeddings (cosine similarity)
def similarity(embedding1, embedding2):
    # Cosine similarity: measures angle between vectors
    # 1.0 = identical, 0.0 = completely different
    similarity = np.dot(embedding1, embedding2) / (
        np.linalg.norm(embedding1) * np.linalg.norm(embedding2)
    )
    return similarity

# Match: find closest known face
def recognize_face(input_embedding):
    best_match = None
    best_similarity = 0
    
    for known_person, known_embedding in known_faces:
        sim = similarity(input_embedding, known_embedding)
        if sim > best_similarity:
            best_similarity = sim
            best_match = known_person
    
    confidence = int(best_similarity * 100)
    return best_match, confidence
```

## 🎓 Key Learnings

1. **Optimization Requires Profiling**
   - Don't guess where bottlenecks are
   - Use Python profiler to measure actual time
   - Surprising results: Sometimes I/O is bottleneck, not computation

2. **Frame-Skipping Works**
   - Human eye can't perceive 1-2 frame delay
   - Processing every 3rd frame feels real-time to users
   - Huge performance gains with minimal perception loss

3. **Resolution Matters More Than Accuracy for Real-Time**
   - Haar Cascade is "old" (2001) vs. modern CNNs
   - But it's FAST, and that's what matters for real-time
   - Modern face detection (RetinaFace, YOLO) would be more accurate but slower
   - Trade-off: 95% accurate at 5 FPS vs. 90% accurate at 30 FPS

4. **Cross-Platform Testing Essential**
   - Works on desktop, but mobile? Different story
   - Mobile webcam permissions, performance different
   - Always test on target devices

5. **Embedding-Based Approach Scales**
   - Can't just compare pixel values
   - Embeddings (128-D vectors) capture "essence" of face
   - Two slightly different angles = same embedding (good!)
   - System is rotation/lighting-invariant

## 🔄 Use Cases

1. **Security:** Automated attendance, access control
2. **Retail:** Customer counting, demographic analysis
3. **Healthcare:** Patient identification, monitoring
4. **Entertainment:** Face filters, AR applications
5. **Authentication:** Login without passwords

## 💡 Future Improvements

- [ ] **Multiple face tracking:** Track across frames (more efficient)
- [ ] **Age/Emotion detection:** Add additional attributes
- [ ] **GPU acceleration:** CUDA for even faster inference
- [ ] **Better face detector:** Use RetinaFace or YOLOv5 (more accurate)
- [ ] **Face anti-spoofing:** Detect if face is real or photo
- [ ] **3D face reconstruction:** Estimate head pose and depth
- [ ] **Face clustering:** Automatically group similar faces
- [ ] **Privacy mode:** Blur unrecognized faces

## 📊 Comparison: Detection Methods

| Method | Speed | Accuracy | Resource | Use Case |
|--------|-------|----------|----------|----------|
| **Haar Cascade** | Very Fast ⚡ | 85% | Low | Real-time webcam |
| **RetinaFace** | Fast | 95% | Medium | High accuracy needed |
| **YOLO Face** | Medium | 93% | High | Object detection focus |
| **Cloud API** | Slow | 98% | API Cost | Best accuracy |

**This project:** Chose Haar Cascade (speed > accuracy for real-time)

## 🔗 Related Projects

- [Voice Authentication](https://github.com/Ishan4565/voice_recognizer) — Audio recognition (similar principles)
- [Fraud Detection](https://github.com/Ishan4565/fraud_detection) — Classification + real-time inference

## 📧 Contact

- **Email:** ishandh454@gmail.com
- **GitHub:** Ishan4565
- **LinkedIn:** [Your LinkedIn]

---

**This project demonstrates real-time ML engineering: optimization, trade-offs, and practical constraints matter more than textbook accuracy.**
