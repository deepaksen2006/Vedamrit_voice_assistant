import sounddevice as sd
import numpy as np

print("Recording starts now. Speak for 5 seconds...")
audio = sd.rec(
    int(5 * 16000),
    samplerate=16000,
    channels=1,
    dtype="float32"
)
sd.wait()

rms = float(np.sqrt(np.mean(audio ** 2)))
print("Microphone RMS:", round(rms, 6))

if rms < 0.003:
    print("WARNING: Very little audio detected.")
else:
    print("Audio detected successfully.")
