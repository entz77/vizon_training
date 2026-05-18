FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime

WORKDIR /app

# System libs required by OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies before copying source so this layer is cached
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy and install the package
COPY . .
RUN pip install --no-cache-dir -e .

# Persistent output directories
RUN mkdir -p logs runs models datasets

VOLUME ["/app/datasets", "/app/runs", "/app/logs", "/app/models"]

CMD ["python", "train.py"]
