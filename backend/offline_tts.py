
import torch
import sounddevice as sd
from transformers import VitsModel, AutoTokenizer

MODEL_NAME = "facebook/mms-tts-hin"

print("Loading Hindi voice model...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = VitsModel.from_pretrained(MODEL_NAME)
model.eval()

print("Hindi voice ready!")

def speak(text):
    if not text.strip():
        return

    inputs = tokenizer(text, return_tensors="pt")

    with torch.no_grad():
        waveform = model(**inputs).waveform

    audio = waveform.squeeze().cpu().numpy()

    sd.play(
        audio,
        samplerate=model.config.sampling_rate
    )
    sd.wait()
