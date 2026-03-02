# Step 1: Use a stable Python base
FROM python:3.11-slim

# Step 2: Set the working directory
WORKDIR /app

# Step 3: Install essential Linux tools (Updated for 2026)
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    g++ \
    libgl1 \
    libglx-mesa0 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Step 4: Copy your project files from GitHub to the server
COPY . .

# Step 5: Install Python libraries (This is where dlib builds)
RUN pip install --no-cache-dir -r requirements.txt

# Step 6: Start the Streamlit app
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
