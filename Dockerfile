FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

COPY . .

RUN pip install --no-cache-dir torchaudio --index-url https://download.pytorch.org/whl/cu121

RUN pip install --no-cache-dir voxcpm

RUN pip install --no-cache-dir \
    "fastapi>=0.104.0" \
    "uvicorn[standard]>=0.24.0" \
    python-multipart \
    soundfile \
    requests

EXPOSE 8000

CMD ["uvicorn", "backend.cloud_server:app", "--host", "0.0.0.0", "--port", "8000"]
