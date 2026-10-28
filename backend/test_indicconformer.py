
import time
import wave
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel


MODEL_ID = "ai4bharat/indic-conformer-600m-multilingual"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
AUDIO_PATH = PROJECT_ROOT / "microphone_test.wav"

EXPECTED_TEXT = (
    "नमस्ते मेरा नाम दीपक सेन है "
    "और मैं वेदअमृत का परीक्षण कर रहा हूँ"
)


def load_audio(path):
    if not path.exists():
        raise FileNotFoundError(
            f"Audio not found: {path}"
        )

    with wave.open(str(path), "rb") as file:
        sample_rate = file.getframerate()
        channels = file.getnchannels()
        sample_width = file.getsampwidth()

        if sample_rate != 16000:
            raise ValueError("Audio must be 16000 Hz")

        if channels != 1:
            raise ValueError("Audio must be mono")

        if sample_width != 2:
            raise ValueError("Audio must be 16-bit PCM")

        raw = file.readframes(file.getnframes())

    audio = np.frombuffer(
        raw, dtype=np.int16
    ).astype(np.float32) / 32768.0

    return torch.from_numpy(audio).unsqueeze(0)


def main():
    print("\nVEDAMRIT INDICCONFORMER TEST")
    print("Audio:", AUDIO_PATH)
    print("Expected:", EXPECTED_TEXT)

    audio = load_audio(AUDIO_PATH)

    print("\nLoading IndicConformer...")
    start_load = time.perf_counter()

    model = AutoModel.from_pretrained(
        MODEL_ID,
        trust_remote_code=True
    )

    model.eval()

    print(
        "Loading time:",
        round(time.perf_counter() - start_load, 2),
        "seconds"
    )

    for decoder in ["ctc", "rnnt"]:
        print("\nTesting decoder:", decoder)

        start = time.perf_counter()

        with torch.inference_mode():
            result = model(
                audio,
                "hi",
                decoder
            )

        elapsed = time.perf_counter() - start

        print("Recognized:", result)
        print("Processing time:", round(elapsed, 2), "seconds")

    print("\nPrevious Whisper Medium result:")
    print(
        "मेरा नाम दीपक सेन है और में "
        "वेद अम्रित का परिक्षण कर रहा है।"
    )

    print("\nTest completed!")


if __name__ == "__main__":
    main()
