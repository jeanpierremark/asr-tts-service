FROM python:3.11-slim

WORKDIR /app

# Dépendances système
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    gcc \
    libsndfile1 \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

    
RUN apt-get update && apt-get install -y ffmpeg

# Copier requirements en premier pour profiter du cache Docker
COPY requirements.txt .

# Installer torch depuis le wheel officiel 
RUN pip install --no-cache-dir torch==2.9.0 --index-url https://download.pytorch.org/whl/cpu

RUN pip install git+https://github.com/huggingface/parler-tts.git


# Installer les autres dépendances
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 7860

CMD ["python", "asr-tts_service.py"]