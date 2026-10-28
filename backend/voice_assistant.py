
import re
import unicodedata

import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel
from ollama import chat

from offline_tts import speak


# ==========================================
# 1. CONFIGURATION
# ==========================================

SAMPLE_RATE = 16000
DURATION = 8

WHISPER_MODEL = "medium"
OLLAMA_MODEL = "qwen3:1.7b"

# True = keyboard testing without microphone or TTS
# False = normal voice assistant
TEXT_TEST_MODE = False

# Set True to compare transcription without VAD
USE_VAD = False

# Session-only user profile
user_profile = {
    "name": None
}

# Keep empty for general multi-user support.
# Add corrections only when appropriate.
NAME_CORRECTIONS = {}


SYSTEM_PROMPT = """
You are VedAmrit (वेदअमृत), a friendly Hindi AI assistant.

Identity:
- Your name is VedAmrit.
- You are an AI assistant.
- Never confuse your name with the user's name.

Conversation:
- Respond in simple, natural Hindi.
- Keep answers short, usually 1-2 sentences.
- Answer the user's current question.
- Do not invent personal information.
- If the question is unclear, ask for clarification.
- Avoid repeating unrelated previous answers.

Medical safety:
- Do not invent medical facts.
- Do not prescribe medicines or dosages.
- Verified Ayurvedic knowledge is not connected yet.
"""


conversation_history = [
    {
        "role": "system",
        "content": SYSTEM_PROMPT
    }
]

whisper = None


# ==========================================
# 2. LOAD WHISPER
# ==========================================

def load_whisper():
    global whisper

    if whisper is None:
        print("\nLoading VedAmrit Speech Recognition...")

        whisper = WhisperModel(
            WHISPER_MODEL,
            device="cpu",
            compute_type="int8"
        )

        print("Speech recognition ready!")


# ==========================================
# 3. RECORD MICROPHONE
# ==========================================

def record_voice():
    print("\nSpeak now...")

    audio = sd.rec(
        int(SAMPLE_RATE * DURATION),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32"
    )

    sd.wait()

    audio = audio.flatten()

    rms = float(
        np.sqrt(np.mean(audio ** 2))
    )

    print(f"Microphone RMS: {rms:.5f}")

    return audio


# ==========================================
# 4. SPEECH TO TEXT
# ==========================================

def recognize_voice(audio):
    if whisper is None:
        load_whisper()

    audio = np.asarray(
        audio,
        dtype=np.float32
    ).flatten()

    if audio.size == 0:
        return ""

    rms = float(
        np.sqrt(np.mean(audio ** 2))
    )

    if rms < 0.003:
        print("Audio level too low.")
        return ""

    segments, info = whisper.transcribe(
        audio,
        language="hi",
        task="transcribe",
        beam_size=5,
        vad_filter=USE_VAD,
        condition_on_previous_text=False,
        no_speech_threshold=0.6,
        initial_prompt=(
            "यह हिंदी में बातचीत है। "
            "सहायक का नाम वेदअमृत है।"
        )
    )

    recognized_text = " ".join(
        segment.text.strip()
        for segment in segments
    ).strip()

    return recognized_text


# ==========================================
# 5. TEXT NORMALIZATION
# ==========================================

def normalize(text):
    return unicodedata.normalize(
        "NFKC",
        str(text)
    ).strip()


def compact_text(text):
    text = normalize(text).lower()

    return re.sub(
        r"[\s।.!?,;:]+",
        "",
        text
    )


# ==========================================
# 6. INTENT DETECTION
# ==========================================

def detect_intent(text):
    compact = compact_text(text)

    # User asks their own name
    if re.search(
        r"(?:मेरा|मेरे|हमारा|हमारे)"
        r"नाम(?:क्या|बताओ|क्याहै)",
        compact
    ):
        return "ask_user_name"

    # User asks assistant's name
    if re.search(
        r"(?:तुम्हारा|तुमारा|तुमहारा|आपका|तेरा)"
        r"नाम(?:क्या|बताओ|क्याहै)",
        compact
    ):
        return "ask_assistant_name"

    # User asks how assistant is
    if re.search(
        r"(?:कैसेहो|कैसीहो|कैसेहैं)",
        compact
    ):
        return "how_are_you"

    return "general"


# ==========================================
# 7. NAME EXTRACTION
# ==========================================

def extract_name(text):
    text = normalize(text)

    if detect_intent(text) != "general":
        return None

    match = re.search(
        r"(?:मेरा|मेरे)\s+नाम\s+"
        r"([\u0900-\u097F]+"
        r"(?:\s+[\u0900-\u097F]+){0,2})"
        r"\s+(?:है|हैं)(?:[।.!?,\s]|$)",
        text
    )

    if not match:
        return None

    name = match.group(1).strip()

    forbidden = {
        "क्या",
        "कौन",
        "कैसे",
        "कब",
        "कहाँ",
        "बताओ",
        "प्रकृति",
        "आहार",
        "तुम्हारा",
        "आपका"
    }

    if any(
        word in forbidden
        for word in name.split()
    ):
        return None

    return NAME_CORRECTIONS.get(
        name,
        name
    )


# ==========================================
# 8. YES / NO RECOGNITION
# ==========================================

def confirmation_tokens(text):
    text = normalize(text).lower()

    text = re.sub(
        r"[।.!?,;:]+",
        " ",
        text
    )

    return text.split()


def is_yes(text):
    tokens = confirmation_tokens(text)

    yes_words = {
        "हाँ",
        "हां",
        "हा",
        "आँ",
        "आं",
        "जी",
        "सही",
        "बिल्कुल",
        "yes",
        "yeah",
        "यस",
        "येस"
    }

    return bool(tokens) and all(
        word in yes_words
        for word in tokens
    )


def is_no(text):
    tokens = confirmation_tokens(text)

    no_words = {
        "नहीं",
        "नही",
        "ना",
        "नो",
        "no",
        "गलत"
    }

    return bool(tokens) and all(
        word in no_words
        for word in tokens
    )


# ==========================================
# 9. CONVERSATION MEMORY
# ==========================================

def remember_exchange(question, answer):
    conversation_history.append({
        "role": "user",
        "content": question
    })

    conversation_history.append({
        "role": "assistant",
        "content": answer
    })

    # System prompt + last 8 exchanges
    if len(conversation_history) > 17:
        del conversation_history[1:-16]


# ==========================================
# 10. GENERATE AI RESPONSE
# ==========================================

def ask_vedamrit(question):
    question = normalize(question)

    intent = detect_intent(question)

    if intent == "ask_assistant_name":
        answer = "मेरा नाम वेदअमृत है।"

    elif intent == "ask_user_name":
        if user_profile["name"]:
            answer = (
                f"आपका नाम "
                f"{user_profile['name']} है।"
            )
        else:
            answer = (
                "आपने अभी तक अपना नाम "
                "नहीं बताया है।"
            )

    elif intent == "how_are_you":
        answer = (
            "मैं ठीक हूँ, धन्यवाद! "
            "आप कैसे हैं?"
        )

    else:
        messages = conversation_history + [
            {
                "role": "system",
                "content": (
                    "Your name is VedAmrit. "
                    "The user's confirmed name is "
                    f"{user_profile['name'] or 'unknown'}. "
                    "Never invent personal details."
                )
            },
            {
                "role": "user",
                "content": question
            }
        ]

        response = chat(
            model=OLLAMA_MODEL,
            messages=messages,
            think=False,
            options={
                "temperature": 0.2,
                "num_predict": 120,
                "num_ctx": 2048
            }
        )

        answer = response["message"]["content"].strip()

        if not answer:
            answer = (
                "माफ कीजिए, "
                "कृपया दोबारा पूछिए।"
            )

    remember_exchange(
        question,
        answer
    )

    return answer


# ==========================================
# 11. OFFLINE TEXT TO SPEECH
# ==========================================

def speak_answer(answer):
    print("\nVedAmrit:", answer)

    if TEXT_TEST_MODE:
        return

    try:
        print("VedAmrit is speaking...")
        speak(answer)

    except Exception as error:
        print("TTS Error:", error)


# ==========================================
# 12. NAME CONFIRMATION
# ==========================================

def handle_name_confirmation(name):
    answer = (
        f"मैंने आपका नाम {name} सुना है। "
        "क्या यह सही है?"
    )

    speak_answer(answer)

    if TEXT_TEST_MODE:
        confirmation = input(
            "Type yes or no: "
        ).strip()

    else:
        input(
            "\nPress Enter to confirm by voice: "
        )

        audio = record_voice()
        confirmation = recognize_voice(audio)

        print("\nYou:", confirmation)

        if not (
            is_yes(confirmation)
            or is_no(confirmation)
        ):
            print(
                "\nVoice confirmation not understood."
            )

            confirmation = input(
                "Type yes or no to confirm: "
            ).strip()

    if is_yes(confirmation):
        user_profile["name"] = name

        answer = (
            f"धन्यवाद! "
            f"मैं आपको {name} कहूँगा।"
        )

    elif is_no(confirmation):
        print("\nRecognized name:", name)

        corrected_name = input(
            "Enter your correct name "
            "(or press Enter to try speaking again): "
        ).strip()

        if corrected_name:
            verify = input(
                f"Save '{corrected_name}' as your name? "
                "(yes/no): "
            ).strip()

            if is_yes(verify):
                user_profile["name"] = corrected_name

                answer = (
                    f"धन्यवाद! "
                    f"मैं आपको {corrected_name} कहूँगा।"
                )
            else:
                answer = (
                    "नाम सेव नहीं किया गया।"
                )

        else:
            answer = (
                "माफ कीजिए, मैंने नाम गलत सुना। "
                "कृपया अपना नाम दोबारा बताइए।"
            )

    else:
        answer = (
            "नाम की पुष्टि नहीं हुई। "
            "कृपया दोबारा कोशिश करें।"
        )

    remember_exchange(
        f"Name confirmation: {confirmation}",
        answer
    )

    speak_answer(answer)


# ==========================================
# 13. TEXT TEST MODE
# ==========================================

def run_text_tests():
    print("\nVedAmrit Text Test Mode")
    print("Type 'exit' to quit.")

    while True:
        question = input(
            "\nYou: "
        ).strip()

        if question.lower() == "exit":
            break

        if not question:
            continue

        name = extract_name(question)

        if name:
            handle_name_confirmation(name)
            continue

        try:
            answer = ask_vedamrit(question)
            speak_answer(answer)

        except Exception as error:
            print("Error:", error)


# ==========================================
# 14. MAIN VOICE ASSISTANT
# ==========================================

def main():
    print("\n================================")
    print("     VEDAMRIT VOICE ASSISTANT")
    print("================================")

    if TEXT_TEST_MODE:
        run_text_tests()
        return

    load_whisper()

    print("\nLanguage: Hindi")
    print("Speech Model:", WHISPER_MODEL)
    print("AI Model:", OLLAMA_MODEL)
    print("Press Enter to speak")
    print("Type 'exit' to quit")

    while True:
        try:
            command = input(
                "\nPress Enter to speak "
                "(exit to quit): "
            ).strip()

            if command.lower() == "exit":
                print("\nVedAmrit stopped.")
                break

            audio = record_voice()

            question = recognize_voice(audio)

            print("\nYou:", question)

            if not question:
                print(
                    "No speech detected. Try again."
                )
                continue

            name = extract_name(question)

            if name:
                handle_name_confirmation(name)
                continue

            answer = ask_vedamrit(question)

            speak_answer(answer)

        except KeyboardInterrupt:
            print("\nVedAmrit stopped.")
            break

        except Exception as error:
            print("\nError:", error)


# ==========================================
# 15. START PROGRAM
# ==========================================

if __name__ == "__main__":
    main()
