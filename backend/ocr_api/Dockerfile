FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    VIRTUAL_ENV=/app/.venv310 \
    PATH="/app/.venv310/bin:$PATH"

WORKDIR /app

# System deps for Pillow / requests + Chromium for zrzuty ekranu stron
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    ca-certificates \
    chromium \
    chromium-driver \
    fonts-liberation \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv ${VIRTUAL_ENV}

# Używamy kontekstu build z katalogu repozytorium (polecenie: docker build -f ocr_api/Dockerfile -t ocr-api-local .)
COPY ocr_api/requirements.txt ./requirements.txt
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

COPY ocr_api/manage.py ./manage.py
COPY ocr_api/ocr_api ./ocr_api
COPY ocr_api/rest_api ./rest_api
# Główny skrypt OCR
COPY ocr_table.py ./ocr_table.py

EXPOSE 8000

# Dłuższy timeout (modele pobierają się i ładują ponad domyślne 30s)
CMD ["gunicorn", "ocr_api.wsgi:application", "--bind", "0.0.0.0:8000", "--workers", "1", "--threads", "1", "--timeout", "600", "--graceful-timeout", "600"]
