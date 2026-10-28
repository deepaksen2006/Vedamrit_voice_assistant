import sounddevice as sd
import soundfile as sf
import numpy as np

SAMPLE_RATE = 16000
DURATION = 8

print("Default microphone:", sd.query_devices(kind="input"))
print("\nRecording starts now. Speak clearly...")

audio = sd.rec(
    int(DURATION * SAMPLE_RATE),
    samplerate=SAMPLE_RATE,
    channels=1,
    dtype="float32"
)

sd.wait()

audio = audio.flatten()

print("Peak audio level:", np.max(np.abs(audio)))
print("RMS audio level:", np.sqrt(np.mean(audio ** 2)))

sf.write("recorded_audio.wav", audio, SAMPLE_RATE)

print("Recording saved successfully!")
