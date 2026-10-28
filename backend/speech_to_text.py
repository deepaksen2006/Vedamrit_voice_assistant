
import soundfile as sf
from faster_whisper import WhisperModel

print("Loading speech recognition model...")

model = WhisperModel(
    "small",
    device="cpu",
    compute_type="int8"
)

print("Model loaded successfully!")

# Load WAV audio directly using soundfile
audio, sample_rate = sf.read(
    "recorded_audio.wav",
    dtype="float32"
)

print("Audio loaded successfully!")
print("Sample rate:", sample_rate)

if sample_rate != 16000:
    raise ValueError("Audio must be recorded at 16000 Hz")

# Convert stereo audio to mono
if audio.ndim == 2:
    audio = audio.mean(axis=1)

# Pass audio samples directly to Whisper
segments, info = model.transcribe(
    audio,
    language="hi",
    task="transcribe",
    beam_size=5,
    vad_filter=True,
    condition_on_previous_text=False
)

text = " ".join(
    segment.text.strip()
    for segment in segments
)

print("\nDetected language:", info.language)
print("Recognized speech:", text)
