# --- Build stage ---
FROM python:3.11-slim AS builder

# Install uv for fast dependency resolution
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app
COPY pyproject.toml ./

# Install dependencies into a virtual env (no editable install yet)
RUN uv venv /opt/venv && \
    uv pip install --python /opt/venv/bin/python \
    torch --index-url https://download.pytorch.org/whl/cpu && \
    uv pip install --python /opt/venv/bin/python \
    openslide-python tifffile numpy pydantic pyyaml loguru typer tqdm \
    opencv-python-headless Pillow

# --- Runtime stage ---
FROM python:3.11-slim

# OpenSlide system library (required by openslide-python)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libopenslide0 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /app
COPY . .

# Volumes for data I/O — mount your WSI files and model here
VOLUME ["/data/wsi", "/data/outputs", "/app/models"]

ENTRYPOINT ["python", "main.py"]
CMD ["--help"]
