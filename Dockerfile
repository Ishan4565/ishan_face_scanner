FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    libgl1 \
    libglx-mesa0 \
    libglib2.0-0 \
    libgomp1 \
    libx11-6 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Upgrade pip first, then install dlib-bin BEFORE face-recognition
RUN pip install --upgrade pip
RUN pip install --no-cache-dir dlib-bin
RUN pip install --no-cache-dir face-recognition
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
