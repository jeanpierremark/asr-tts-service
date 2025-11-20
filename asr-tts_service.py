import base64
from flask import Flask, request, jsonify, send_file
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline, SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
import torch
import soundfile as sf
import numpy as np
import io

app = Flask(__name__)

# ASR wolof
asr = pipeline("automatic-speech-recognition", model="bilalfaye/wav2vec2-large-mms-1b-wolof")

# Model traduction
model_name = "MaroneAI/nllb-Wolof-to-French-615M"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
translation_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# TTS wolof
tts_processor = SpeechT5Processor.from_pretrained("Moustapha91/TTS_WOLOF_FINAL")
tts_model = SpeechT5ForTextToSpeech.from_pretrained("Moustapha91/TTS_WOLOF_FINAL")
tts_vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")

# Speaker embedding
speaker_embedding = torch.randn(1, 512)


@app.route("/accueil", methods=["GET"])
def accueil():
    return "Flask ASR + TTS Service is running!"


# Traduction Wolof → Français
def wolofToFrench(wolof_text):
    src_lang = "wol_Latn"
    tgt_lang = "fra_Latn"

    tokenizer.src_lang = src_lang
    inputs = tokenizer(wolof_text, return_tensors="pt", padding=True)

    forced_bos = tokenizer.convert_tokens_to_ids(tgt_lang)
    translated_tokens = translation_model.generate(
        **inputs,
        forced_bos_token_id=forced_bos,
        max_new_tokens=200
    )

    translated_text = tokenizer.batch_decode(
        translated_tokens,
        skip_special_tokens=True
    )[0]

    return translated_text


# ASR  Traduction
@app.route("/transcribe", methods=["POST"])
def transcribe():
    if "file" not in request.files:
        return jsonify({"error": "Aucun fichier 'file' trouvé"}), 400

    audio_file = request.files["file"]
    data, samplerate = sf.read(audio_file)

    if data is None or len(data) == 0:
        return jsonify({"error": "Fichier audio vide ou invalide"}), 400

    text = asr(np.array(data))["text"]
    print(text)

    translated = wolofToFrench(text)
    print(translated)

    return translated if translated.strip() else "Bonjour"


# TTS
@app.route("/tts", methods=["POST"])
def tts_route():
    data = request.get_json()

    if not data or "text" not in data:
        return jsonify({"error": "Champ 'text' manquant"}), 400

    text = data["text"]

    # Durée estimée
    words = len(text.split())
    duration_seconds = words / 2.5
    max_samples = int(duration_seconds * 16000)

    inputs = tts_processor(text=text, return_tensors="pt")

    speech = tts_model.generate_speech(
        inputs["input_ids"],
        speaker_embedding,
        vocoder=tts_vocoder
    )

    audio = speech.numpy()[:max_samples]

    buffer = io.BytesIO()
    sf.write(buffer, audio, 16000, format='WAV')
    buffer.seek(0)

    audio_b64 = base64.b64encode(buffer.read()).decode("utf-8")

    return jsonify({
        "audio": audio_b64,
        "duration_seconds": duration_seconds,
        "samples": len(audio)
    })


if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=5002, use_reloader=False)
