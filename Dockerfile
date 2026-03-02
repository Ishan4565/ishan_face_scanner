# Step 1: Use a version of Python that plays nice with dlib
FROM python:3.11-slim

# Step 2: Set the working directory inside the container
WORKDIR /app

# Step 3: Install the "Missing" Linux tools (C++ compiler and CMake)
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    g++ \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Step 4: Copy your project files into the container
COPY . .

# Step 5: Install Python libraries (dlib will now find CMake!)
RUN pip install --no-cache-dir -r requirements.txt

# Step 6: Tell the server how to run your app
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
