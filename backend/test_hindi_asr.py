
import os
import time
import wave

import numpy as np
from faster_whisper import WhisperModel


# =====================================
# CONFIGURATION
# =====================================

AUDIO_FILE = "microphone_test.wav"
MODELS = ["small", "medium"]
EXPECTED_TEXT = (
    "नमस्ते मेरा नाम दीपक सेन है "
    "और मैं वेदअमृत का परीक्षण कर रहा हूँ"
)


# =====================================
# LOAD WAV FILE
# =====================================

def load_audio(path):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Audio file not found: {path}"
        )

    with wave.open(path, "rb") as wav:
        sample_rate = wav.getframerate()
        channels = wav.getnchannels()
        sample_width = wav.getsampwidth()
        raw_audio = wav.readframes(
            wav.getnframes()
        )

    if sample_rate != 16000:
        raise ValueError(
            f"Expected 16000 Hz, got {sample_rate}"
        )

    if channels != 1:
        raise ValueError(
            "Expected mono audio"
        )

    if sample_width != 2:
        raise ValueError(
            "Expected 16-bit PCM WAV"
        )

    audio = np.frombuffer(
        raw_audio,
        dtype=np.int16
    ).astype(np.float32) / 32768.0

    return audio


# =====================================
# TEST ONE MODEL
# =====================================

def test_model(model_name, audio):
    print("\n" + "=" * 55)
    print("Testing:", model_name)
    print("=" * 55)

    model = WhisperModel(
        model_name,
        device="cpu",
        compute_type="int8"
    )

    start = time.perf_counter()

    segments, info = model.transcribe(
        audio,
        language="hi",
        task="transcribe",
        beam_size=5,
        vad_filter=True,
        condition_on_previous_text=False
    )

    recognized_text = " ".join(
        segment.text.strip()
        for segment in segments
    ).strip()

    elapsed = time.perf_counter() - start

    print("Expected:", EXPECTED_TEXT)
    print("Recognized:", recognized_text)
    print("Detected language:", info.language)
    print("Processing time:", round(elapsed, 2), "seconds")

    return recognized_text


# =====================================
# MAIN
# =====================================

def main():
    print("\nVEDAMRIT HINDI ASR TEST")

    audio = load_audio(AUDIO_FILE)

    print("Audio file:", AUDIO_FILE)
    print("Audio duration:", round(len(audio) / 16000, 2), "sec")
    print("Audio samples:", len(audio))

    results = {}

    for model_name in MODELS:
        results[model_name] = test_model(
            model_name,
            audio
        )

    print("\n" + "=" * 55)
    print("FINAL COMPARISON")
    print("=" * 55)

    for model_name, result in results.items():
        print(f"{model_name}: {result}")

    print("\nTest completed!")


if __name__ == "__main__":
    main()
