FROM python:3.11-slim

WORKDIR /app

# Install only the absolute minimum system tools
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    libgl1 \
    libglx-mesa0 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY . .

# Set a limit on the compiler memory
ENV MAKEFLAGS="-j1"

# Install requirements (using dlib-bin from your requirements.txt)
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
