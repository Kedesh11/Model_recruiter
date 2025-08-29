# SEEG Dashboard - Dockerfile
FROM python:3.11-slim

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# System deps (OCR + common libs for imaging)
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    tesseract-ocr-fra \
    libglib2.0-0 libsm6 libxrender1 libxext6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first for better layer caching
COPY requirements.txt ./
RUN pip install --upgrade pip && \
    pip install -r requirements.txt && \
    python -m spacy download fr_core_news_md

# Copy the rest of the code
COPY . .

# Runtime configuration must be provided at run-time (docker run --env-file or docker compose)
# Intentionally avoid baking secrets (SUPABASE_URL, SUPABASE_KEY, OPENAI_API_KEY) into the image
ENV OPENAI_MODEL="gpt-4o-mini" \
    VISION_FILE="/app/scripts/vision_seeg.md"

EXPOSE 8501

# Run Streamlit app
CMD ["streamlit", "run", "streamlit_app/app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true", "--browser.gatherUsageStats=false", "--server.enableCORS=false"]
