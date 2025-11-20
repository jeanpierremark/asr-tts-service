FROM python:3.13-slim

WORKDIR /app

# Dépendances système
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copier requirements en premier pour profiter du cache Docker
COPY requirements.txt .

# Installer torch depuis le wheel officiel 
RUN pip install --no-cache-dir torch==2.9.1 --index-url https://download.pytorch.org/whl/cpu

# Installer les autres dépendances
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 5002

CMD ["python", "asr-tts_service.py"]
