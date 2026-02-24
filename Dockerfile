# MVDiffusion inference service: FastAPI + queue worker
# Build: docker build -t mvdiffusion-inference:latest .
# Run: docker run --env-file .env -p 9000:9000 -v ./weights:/app/weights -v ./outputs:/app/outputs mvdiffusion-inference:latest

FROM python:3.10-slim

WORKDIR /app

# System deps if needed (e.g. for PyTorch/CUDA in a full image)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/
COPY configs/ ./configs/
COPY src/ ./src/
COPY generate_video_tool/ ./generate_video_tool/
# demo.py and weights/ are optional for placeholder; mount weights at runtime
COPY demo.py .

# Default env (override via -e or --env-file)
ENV HTTP_HOST=0.0.0.0
ENV HTTP_PORT=9000
ENV REDIS_URL=redis://localhost:6379/0
ENV TASK_QUEUE=panorama:task
ENV RESULT_QUEUE=panorama:result

EXPOSE 9000

# Use env at runtime: -e HTTP_HOST=0.0.0.0 -e HTTP_PORT=9000 or --env-file
CMD ["sh", "-c", "exec python -m uvicorn app.main:app --host ${HTTP_HOST:-0.0.0.0} --port ${HTTP_PORT:-9000}"]
