# RetinaLens backend (FastAPI + ONNX model).
# Works on Hugging Face Spaces (Docker, port 7860, non-root user), Railway, Fly.io and Render.
# Needs about 1 GB of RAM. The frontend is deployed separately on Vercel (see vercel.json).
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# OpenCV runtime libraries
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Hugging Face Spaces run containers as user 1000; give it a home and a writable app folder.
RUN useradd -m -u 1000 user
ENV HOME=/home/user \
    HF_HOME=/home/user/.cache/huggingface
WORKDIR /app

# CPU-only PyTorch first (much smaller than the default CUDA build), then the rest.
COPY requirements.txt .
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch==2.14.0 torchvision==0.29.0 \
    && pip install -r requirements.txt

COPY --chown=user backend ./backend
COPY --chown=user ml ./ml
COPY --chown=user checkpoints ./checkpoints
RUN mkdir -p /app/storage && chown -R user /app
USER user

# Set GEMINI_API_KEY as a secret in the hosting dashboard (never bake secrets into the image).
# PORT is set by Render/Railway/Fly; Hugging Face Spaces expects 7860.
EXPOSE 7860
CMD ["sh", "-c", "uvicorn backend.main:app --host 0.0.0.0 --port ${PORT:-7860}"]
