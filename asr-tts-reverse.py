import base64
import io
import os
import re
import logging
import numpy as np
import soundfile as sf
import torch
import google.generativeai as genai
import requests  
import tempfile 
from flask import Flask, request, jsonify
from transformers import pipeline, AutoTokenizer
from parler_tts import ParlerTTSForConditionalGeneration
from pydub import AudioSegment  
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor

# Charger les variables d'environnement
load_dotenv()

# Configuration Gemini
google_api = os.getenv('GOOGLE_API_KEY')
genai.configure(api_key=google_api)
model_gemini = genai.GenerativeModel('gemini-flash-latest')

# Configuration générale et Optimisations CPU
device = "cpu"
torch.set_num_threads(os.cpu_count()) # Utilise tous les coeurs disponibles
torch.set_grad_enabled(False)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Chargement des modèles
# ASR - Wolof (Local)
asr = pipeline(
    task="automatic-speech-recognition",
    model="bilalfaye/wav2vec2-large-mms-1b-wolof",
    device=-1
)

# TTS - Parler TTS (Adia) avec optimisation de structure
tts_model_id = "CONCREE/Adia_TTS"
tts_model = ParlerTTSForConditionalGeneration.from_pretrained(tts_model_id).to(device)
tts_model.eval()

# Tente d'utiliser l'attention accélérée si disponible
try:
    tts_model = tts_model.to_bettertransformer()
except:
    pass

tts_tokenizer = AutoTokenizer.from_pretrained(tts_model_id)

voice_description = """
    A professional female voice with a smooth, clear, and elegant timbre. 
    The speech is delivered at a moderate, steady pace with a warm and welcoming tone. 
    The articulation is crystal clear and very precise, typical of a high-end telecommunications assistant. 
    The audio is recorded in a silent studio environment, sounding crisp, intimate, and without any breathiness or background noise. 
    The intonation is melodic yet professional, conveying trust and reliability.
    """
description_id = tts_tokenizer(voice_description, return_tensors="pt").input_ids.to(device)

# --- Fonctions Utilitaires ---

UNITS = {0: "zéro", 1: "un", 2: "deux", 3: "trois", 4: "quatre", 5: "cinq", 6: "six", 7: "sept", 8: "huit", 9: "neuf", 10: "dix", 11: "onze", 12: "douze", 13: "treize", 14: "quatorze", 15: "quinze", 16: "seize"}
TENS = {20: "vingt", 30: "trente", 40: "quarante", 50: "cinquante", 60: "soixante", 80: "quatre-vingt"}

def number_to_french(n: int) -> str:
    if n < 17: return UNITS[n]
    if n < 20: return "dix-" + UNITS[n - 10]
    if n < 70:
        tens, unit = divmod(n, 10)
        base = TENS[tens * 10]
        if unit == 0: return base
        if unit == 1: return base + " et un"
        return base + "-" + UNITS[unit]
    if n < 80: return "soixante-" + number_to_french(n - 60)
    if n < 100:
        base = "quatre-vingt"
        if n == 80: return base
        return base + "-" + number_to_french(n - 80)
    hundreds, rest = divmod(n, 100)
    base = "cent" if hundreds == 1 else UNITS[hundreds] + " cent"
    return base if rest == 0 else base + " " + number_to_french(rest)

def convert_digits_in_text(text: str) -> str:
    def repl(match):
        s = match.group(0)
        if set(s) == {"0"}: return " ".join(["zéro"] * len(s))
        zeros = len(s) - len(s.lstrip("0"))
        rest = s.lstrip("0")
        out = " ".join(["zéro"] * zeros)
        return f"{out} {number_to_french(int(rest))}".strip()
    return re.sub(r"\d+", repl, text)

def split_by_sentences(text: str, max_chars: int = 100) -> list:
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks, current = [], ""
    for s in sentences:
        if len(current) + len(s) <= max_chars:
            current = (current + " " + s).strip()
        else:
            if current: chunks.append(current)
            current = s
    if current: chunks.append(current)
    return chunks

def normalize_audio(audio: np.ndarray, peak: float = 0.9) -> np.ndarray:
    m = np.max(np.abs(audio))
    if m > 0: audio = audio * (peak / m)
    return np.clip(audio, -1.0, 1.0)

def smooth_concat(segments, sr, fade_ms=20):
    if not segments: return np.array([], dtype=np.float32)
    if len(segments) == 1: return segments[0]
    fade_len = int(sr * fade_ms / 1000)
    output = segments[0]
    for i in range(1, len(segments)):
        next_seg = segments[i]
        actual_fade = min(fade_len, len(output), len(next_seg))
        if actual_fade > 0:
            fade_out = np.linspace(1.0, 0.0, actual_fade)
            fade_in = np.linspace(0.0, 1.0, actual_fade)
            overlap = (output[-actual_fade:] * fade_out) + (next_seg[:actual_fade] * fade_in)
            output = np.concatenate([output[:-actual_fade], overlap, next_seg[actual_fade:]])
        else:
            output = np.concatenate([output, next_seg])
    return output

# --- Logique TTS Parallélisée ---

def generate_one_chunk(chunk):
    """Génère l'audio pour un seul morceau de texte"""
    if not chunk.strip(): return None
    full_chunk = chunk if chunk.endswith(('.', '!', '?')) else chunk + "."
    prompt_ids = tts_tokenizer(full_chunk, return_tensors="pt").input_ids.to(device)
    with torch.no_grad():
        audio = tts_model.generate(
            input_ids=description_id,
            prompt_input_ids=prompt_ids,
            max_new_tokens=512, # Réduit pour plus de rapidité
            do_sample=True,
            temperature=0.7,
            num_beams=1 # Désactive la recherche coûteuse
        )
    return audio.cpu().numpy().squeeze().astype(np.float32)

def generate_tts_optimized(text: str) -> str:
    text = convert_digits_in_text(text)
    chunks = split_by_sentences(text, max_chars=80) 
    
    # Exécution en parallèle de chaque morceau
    with ThreadPoolExecutor() as executor:
        results = list(executor.map(generate_one_chunk, chunks))
    
    audio_segments = [res for res in results if res is not None and res.size > 0]
    
    if not audio_segments: return ""
    final_audio = smooth_concat(audio_segments, tts_model.config.sampling_rate)
    final_audio = normalize_audio(final_audio)
    
    buffer = io.BytesIO()
    sf.write(buffer, final_audio, tts_model.config.sampling_rate, format="WAV")
    buffer.seek(0)
    return "data:audio/wav;base64," + base64.b64encode(buffer.read()).decode()

# --- Traduction & Routes ---

def french_to_wolof_with_gemini(text: str) -> str:
    prompt = f"""
        Tu es un traducteur expert en wolof travaillant pour la Sen'eau. 
        Traduis le texte suivant du Français vers le wolof. 
        Ne donnes pas d'explication traduis uniquement le texte en wolof.
        Utilise un ton poli, professionnel et garde les termes techniques usuels (Sen'eau, compteur, branchement, Assistante virtuelle ).
        
        Texte : {text}"""
    try:
        response = model_gemini.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return "Naka ngeen def ! Man la ADAMA, seen Assistante virtuelle bu Sen'eau. Ma ngi fi ngir dimbali leen ci seeni laaj yépp yu jëm ci wàllu ndoxum naan ci Sénégal."

def wolof_to_french_gemini(text: str) -> str:
    prompt = f"""
        Tu es un traducteur expert en français travaillant pour la Sen'eau. 
        Traduis le texte suivant du Wolof vers le français. 
        Ne donnes pas d'explication traduis uniquement le texte en français.
        Utilise un ton poli, professionnel et garde les termes techniques usuels (compteur, branchement).
        SI LE TEXTE EST INTELLIGIBLE, n'explique pas pourquoi. Renvoie simplement le texte original tel quel ou renvoies Bonjour.
        Ne fais aucun commentaire, donnes uniquement la traduction
          
        Texte : {text}""" 
    try:
        response = model_gemini.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return text or 'Bonjour'

@app.route("/", methods=["GET"])
def healthcheck():
    return "Service ASR/TTS Sen'eau opérationnel"

@app.route("/transcribe", methods=["POST"])
def transcribe():
    if "file" not in request.files: return "Bonjour", 400
    data, sr = sf.read(request.files["file"])
    data = np.asarray(data, dtype=np.float32)
    if data.ndim > 1: data = data.mean(axis=1)
    wolof_text = asr(normalize_audio(data))["text"]
    return wolof_to_french_gemini(wolof_text)

@app.route("/tts", methods=["POST"])
def tts():
    payload = request.get_json()
    if not payload or "text" not in payload: return jsonify({"error": "Texte manquant"}), 400
    wolof_text = french_to_wolof_with_gemini(payload["text"])
    audio_base64 = generate_tts_optimized(wolof_text)
    return jsonify({"wolof_text": wolof_text, "audio": audio_base64})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860)
