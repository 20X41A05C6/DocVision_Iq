

# -----------------------------
# Base Image
# -----------------------------
FROM python:3.10

# -----------------------------
# Environment
# -----------------------------
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# -----------------------------
# System Dependencies
# -----------------------------
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1 \
    poppler-utils \
    curl \
    && rm -rf /var/lib/apt/lists/*

# -----------------------------
# Working Directory
# -----------------------------
WORKDIR /app

# -----------------------------
# Upgrade pip tools
# -----------------------------
RUN pip install --upgrade pip setuptools wheel

# -----------------------------
# Install PyTorch CPU FIRST
# -----------------------------
RUN pip install --no-cache-dir \
    torch==2.1.2+cpu \
    torchvision==0.16.2+cpu \
    torchaudio==2.1.2+cpu \
    --index-url https://download.pytorch.org/whl/cpu

# -----------------------------
# Copy requirements
# -----------------------------
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# -----------------------------
# Copy project files
# -----------------------------
COPY . .

# -----------------------------
# Runtime directories
# -----------------------------
RUN mkdir -p uploads/images

# -----------------------------
# Hugging Face PUBLIC PORT
# -----------------------------
EXPOSE 7860

# -----------------------------
# Start FastAPI (internal) + Streamlit (public)
# -----------------------------
CMD ["bash", "-c", "uvicorn src.main:app --host 0.0.0.0 --port 8000 & exec streamlit run frontend/app.py --server.port=7860 --server.address=0.0.0.0 --server.headless=true --server.enableXsrfProtection=false"]

