# RetinaLens backend (FastAPI + ONNX Runtime). No PyTorch: the server only runs the exported
# ONNX models, so it fits in about 400 MB of RAM (Render free tier: 512 MB).
# Works on Render (render.yaml), Railway, Fly.io and Hugging Face Spaces (port 7860, user 1000).
# The frontend is deployed separately on Vercel (see vercel.json).
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Hugging Face Spaces run containers as user 1000; give it a home and a writable app folder.
RUN useradd -m -u 1000 user
ENV HOME=/home/user
WORKDIR /app

COPY requirements-server.txt .
RUN pip install -r requirements-server.txt

COPY --chown=user backend ./backend
COPY --chown=user ml ./ml
COPY --chown=user checkpoints ./checkpoints
RUN mkdir -p /app/storage && chown -R user /app
USER user

# Set GEMINI_API_KEY as a secret in the hosting dashboard (never bake secrets into the image).
# PORT is set by Render/Railway/Fly; Hugging Face Spaces expects 7860.
EXPOSE 7860
CMD ["sh", "-c", "uvicorn backend.main:app --host 0.0.0.0 --port ${PORT:-7860}"]
