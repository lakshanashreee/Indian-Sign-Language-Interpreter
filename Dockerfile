FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for OpenCV and MediaPipe
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency requirements
COPY requirements.txt .

# Install Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy application backend, static frontend, and pre-trained models
COPY backend/ ./backend/
COPY static/ ./static/
COPY hand_landmarker.task .
COPY isl_landmark_model.joblib .

# Expose port (7860 default for Hugging Face Spaces, overridable by $PORT on Render/Railway)
ENV PORT=7860
EXPOSE 7860

# Launch Uvicorn ASGI server
CMD ["sh", "-c", "uvicorn backend.main:app --host 0.0.0.0 --port ${PORT:-7860}"]

