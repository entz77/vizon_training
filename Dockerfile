FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime

WORKDIR /app

# System libs + Google Cloud SDK (for gsutil)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    curl \
    gnupg \
    && echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" \
       > /etc/apt/sources.list.d/google-cloud-sdk.list \
    && curl -fsSL https://packages.cloud.google.com/apt/doc/apt-key.gpg \
       | gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg \
    && apt-get update && apt-get install -y --no-install-recommends google-cloud-cli \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies before copying source so this layer is cached
COPY requirements.txt .
RUN pip install --no-cache-dir --default-timeout=300 -r requirements.txt

# Copy and install the package
COPY . .
RUN pip install --no-cache-dir .

ENV PYTHONPATH=/app

# Persistent output directories
RUN mkdir -p logs runs models datasets

VOLUME ["/app/datasets", "/app/runs", "/app/logs", "/app/models"]

CMD ["python", "train.py"]
