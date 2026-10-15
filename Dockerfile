# RetinaLens backend (FastAPI + ONNX model). Deploy on Render, Railway, Fly.io or Hugging Face Spaces.
# The frontend is deployed separately on Vercel (see vercel.json).
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# OpenCV runtime libraries
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# CPU-only PyTorch first (much smaller than the default CUDA build), then the rest.
COPY requirements.txt .
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch==2.14.0 torchvision==0.29.0 \
    && pip install -r requirements.txt

COPY backend ./backend
COPY ml ./ml
COPY checkpoints ./checkpoints

# Set GEMINI_API_KEY in the hosting dashboard (never bake secrets into the image).
EXPOSE 8000
CMD ["sh", "-c", "uvicorn backend.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
