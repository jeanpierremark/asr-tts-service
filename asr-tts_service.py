import base64
import io
import re
import logging
import google.generativeai as genai
import numpy as np
import soundfile as sf
import torch
from flask import Flask, request, jsonify
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from parler_tts import ParlerTTSForConditionalGeneration

# Configuration générale
device = "cpu"
torch.set_grad_enabled(False)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)


# Chargement des modèles
asr = pipeline(
    task="automatic-speech-recognition",
    model="bilalfaye/wav2vec2-large-mms-1b-wolof",
    device=-1
)


tts_model_id = "CONCREE/Adia_TTS"
tts_model = ParlerTTSForConditionalGeneration.from_pretrained(tts_model_id).to(device)
tts_model.eval()
tts_tokenizer = AutoTokenizer.from_pretrained(tts_model_id)

voice_description = """
A calm, professional and reassuring female voice.
Clear articulation with a moderate speaking pace.
Warm and respectful tone suitable for customer service.
Institutional and trustworthy style, adapted to public services.
"""
description_id = tts_tokenizer(voice_description, return_tensors="pt").input_ids.to(device)

# Fonctions utilitaires (Texte & Audio)

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

def clean_text(text: str) -> str:
    text = re.sub(r"[^\w\s.,!?']", "", text)
    return re.sub(r"\s+", " ", text).strip()

def split_by_sentences(text: str, max_chars: int = 150) -> list:
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
    """Combine les segments audio avec un fondu enchaîné propre."""
    if not segments: return np.array([], dtype=np.float32)
    if len(segments) == 1: return segments[0]

    fade_len = int(sr * fade_ms / 1000)
    output = segments[0]

    for i in range(1, len(segments)):
        next_seg = segments[i]
        # On s'assure que le fondu n'est pas plus long que les segments eux-mêmes
        actual_fade = min(fade_len, len(output), len(next_seg))
        
        if actual_fade > 0:
            # Création des courbes de fondu
            fade_out = np.linspace(1.0, 0.0, actual_fade)
            fade_in = np.linspace(0.0, 1.0, actual_fade)
            
            # Application du mélange sur la zone de recouvrement
            overlap = (output[-actual_fade:] * fade_out) + (next_seg[:actual_fade] * fade_in)
            
            # Reconstruction du signal
            output = np.concatenate([output[:-actual_fade], overlap, next_seg[actual_fade:]])
        else:
            output = np.concatenate([output, next_seg])
            
    return output


# Logique TTS et Traduction
def generate_tts_optimized(text: str) -> str:
    text = convert_digits_in_text(text)
    chunks = split_by_sentences(text)
    audio_segments = []

    for chunk in chunks:
        prompt_ids = tts_tokenizer(chunk, return_tensors="pt").input_ids.to(device)

        with torch.no_grad():
            # Correction : max_new_tokens augmenté et do_sample activé pour la stabilité
            audio = tts_model.generate(
                input_ids=description_id,
                prompt_input_ids=prompt_ids,
                max_new_tokens=1024, 
                do_sample=True,
                temperature=1.0
            )

        audio_np = audio.cpu().numpy().squeeze().astype(np.float32)
        if audio_np.ndim == 0: continue # Évite les erreurs sur segments vides
        audio_segments.append(audio_np)

    if not audio_segments:
        return ""

    final_audio = smooth_concat(audio_segments, tts_model.config.sampling_rate)
    final_audio = normalize_audio(final_audio)

    buffer = io.BytesIO()
    sf.write(buffer, final_audio, tts_model.config.sampling_rate, format="WAV")
    buffer.seek(0)

    return "data:audio/wav;base64," + base64.b64encode(buffer.read()).decode()



def french_to_wolof_with_gemini(text: str) -> str:
    prompt = f"""
    Tu es un traducteur expert en Wolof travaillant pour la Sen'eau. 
    Traduis le texte suivant du Français vers le Wolof. 
    Utilise un ton poli, professionnel et garde les termes techniques usuels (compteur, branchement).
    
    Texte : {text}
    """
    model = genai.GenerativeModel('gemini-1.5-flash')
    response = model.generate_content(prompt)
    return response.text

def wolof_to_french_gemini(text: str) -> str:
    prompt = f"""
    Tu es un interprète professionnel Wolof-Français. 
    Traduis le message suivant en français fluide et professionnel. 
    Contexte : Service client de la Sen'eau (eau potable au Sénégal).
    
    Texte en Wolof : {text}
    """
    model = genai.GenerativeModel('gemini-1.5-flash')
    response = model.generate_content(prompt)
    return response.text.strip()

# Routes Flask
@app.route("/", methods=["GET"])
def healthcheck():
    return "ASR / Translation / TTS service running (CPU)"

@app.route("/transcribe", methods=["POST"])
def transcribe():
    if "file" not in request.files: return jsonify({"error": "Fichier audio manquant"}), 400
    data, sr = sf.read(request.files["file"])
    data = np.asarray(data, dtype=np.float32)
    if data.ndim > 1: data = data.mean(axis=1)
    data = normalize_audio(data)
    result = asr(data)
    wolof_text = result["text"]
    french_text = wolof_to_french_gemini(wolof_text)
    return french_text or 'Bonjour'

@app.route("/tts", methods=["POST"])
def tts():
    payload = request.get_json()
    if not payload or "text" not in payload: return jsonify({"error": "Champ 'text' requis"}), 400
    wolof_text = french_to_wolof_with_gemini(payload["text"])
    audio_base64 = generate_tts_optimized(wolof_text)
    return jsonify({"audio": audio_base64})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860)