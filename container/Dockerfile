FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir \
    ultralytics \
    google-genai \
    opencv-python-headless \
    pillow \
    python-dotenv

# Copy shared source modules
COPY src/ src/
COPY yolov8n.pt .

# Copy worker entry point
COPY container/worker.py .

ENV PYTHONPATH=/app

CMD ["python", "worker.py"]
