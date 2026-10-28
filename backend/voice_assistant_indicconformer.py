"""VedAmrit Hindi voice assistant with IndicConformer RNNT ASR."""
import re
import unicodedata

import numpy as np
import sounddevice as sd
import torch
from transformers import AutoModel
from ollama import chat
from offline_tts import speak

SAMPLE_RATE = 16000
DURATION = 8
ASR_MODEL_ID = "ai4bharat/indic-conformer-600m-multilingual"
OLLAMA_MODEL = "qwen3:1.7b"
TEXT_TEST_MODE = False

user_profile = {"name": None}
conversation_history = [{"role": "system", "content": (
    "You are VedAmrit (वेदअमृत), a friendly Hindi AI assistant. "
    "Reply in simple, short Hindi. Never invent personal details. "
    "Do not prescribe medicine or dosages. Verified Ayurvedic knowledge is not connected yet."
)}]
asr_model = None


def load_asr():
    global asr_model
    if asr_model is None:
        print("Loading IndicConformer RNNT (first run may take time)...")
        asr_model = AutoModel.from_pretrained(ASR_MODEL_ID, trust_remote_code=True)
        asr_model.eval()
        print("IndicConformer ready!")


def record_voice():
    print("\nSpeak now...")
    audio = sd.rec(int(SAMPLE_RATE * DURATION), samplerate=SAMPLE_RATE,
                   channels=1, dtype="float32")
    sd.wait()
    audio = audio.flatten()
    print(f"Microphone RMS: {float(np.sqrt(np.mean(audio ** 2))):.5f}")
    return audio


def recognize_voice(audio):
    load_asr()
    audio = np.asarray(audio, dtype=np.float32).flatten()
    if audio.size == 0 or float(np.sqrt(np.mean(audio ** 2))) < 0.003:
        return ""
    # The official model interface expects a batch of mono 16 kHz waveforms.
    waveform = torch.from_numpy(audio.copy()).unsqueeze(0)
    with torch.inference_mode():
        result = asr_model(waveform, "hi", "rnnt")
    if isinstance(result, (list, tuple)):
        return " ".join(str(part) for part in result).strip()
    return str(result).strip()


def normalize(text):
    return unicodedata.normalize("NFKC", str(text)).strip()


def compact_text(text):
    return re.sub(r"[\s।.!?,;:]+", "", normalize(text).lower())


def detect_intent(text):
    compact = compact_text(text)
    if re.search(r"(?:मेरा|मेरे|हमारा|हमारे)नाम(?:क्या|बताओ|क्याहै)", compact):
        return "ask_user_name"
    if re.search(r"(?:तुम्हारा|तुमारा|तुमहारा|आपका|तेरा)नाम(?:क्या|बताओ|क्याहै)", compact):
        return "ask_assistant_name"
    if re.search(r"(?:कैसेहो|कैसीहो|कैसेहैं)", compact):
        return "how_are_you"
    return "general"


def extract_name(text):
    text = normalize(text)
    if detect_intent(text) != "general":
        return None
    match = re.search(
        r"(?:मेरा|मेरे)\s+नाम\s+([\u0900-\u097F]+(?:\s+[\u0900-\u097F]+){0,2})"
        r"\s+(?:है|हैं)(?:[।.!?,\s]|$)", text)
    if not match:
        return None
    name = match.group(1).strip()
    forbidden = {"क्या", "कौन", "कैसे", "कब", "कहाँ", "बताओ", "प्रकृति", "आहार", "तुम्हारा", "आपका"}
    if any(word in forbidden for word in name.split()):
        return None
    return name


def confirmation_tokens(text):
    return re.sub(r"[।.!?,;:]+", " ", normalize(text).lower()).split()


def is_yes(text):
    tokens = confirmation_tokens(text)
    words = {"हाँ", "हां", "हा", "आँ", "आं", "जी", "सही", "बिल्कुल", "yes", "yeah", "यस", "येस"}
    return bool(tokens) and all(t in words for t in tokens)


def is_no(text):
    tokens = confirmation_tokens(text)
    words = {"नहीं", "नही", "ना", "नो", "no", "गलत"}
    return bool(tokens) and all(t in words for t in tokens)


def remember_exchange(question, answer):
    conversation_history.extend([
        {"role": "user", "content": question},
        {"role": "assistant", "content": answer},
    ])
    if len(conversation_history) > 17:
        del conversation_history[1:-16]


def ask_vedamrit(question):
    question = normalize(question)
    intent = detect_intent(question)
    if intent == "ask_assistant_name":
        answer = "मेरा नाम वेदअमृत है।"
    elif intent == "ask_user_name":
        answer = (f"आपका नाम {user_profile['name']} है।" if user_profile["name"]
                  else "आपने अभी तक अपना नाम नहीं बताया है।")
    elif intent == "how_are_you":
        answer = "मैं ठीक हूँ, धन्यवाद! आप कैसे हैं?"
    else:
        messages = conversation_history + [
            {"role": "system", "content": (
                "Your name is VedAmrit. The user's confirmed name is "
                f"{user_profile['name'] or 'unknown'}. Never invent personal details.")},
            {"role": "user", "content": question},
        ]
        response = chat(model=OLLAMA_MODEL, messages=messages, think=False,
                        options={"temperature": 0.2, "num_predict": 120, "num_ctx": 2048})
        answer = response["message"]["content"].strip() or "माफ कीजिए, कृपया दोबारा पूछिए।"
    remember_exchange(question, answer)
    return answer


def speak_answer(answer):
    print("\nVedAmrit:", answer)
    if not TEXT_TEST_MODE:
        try:
            print("VedAmrit is speaking...")
            speak(answer)
        except Exception as exc:
            print("TTS Error:", exc)


def handle_name_confirmation(name):
    speak_answer(f"मैंने आपका नाम {name} सुना है। क्या यह सही है?")
    if TEXT_TEST_MODE:
        confirmation = input("Type yes or no: ").strip()
    else:
        input("\nPress Enter to confirm by voice: ")
        confirmation = recognize_voice(record_voice())
        print("\nYou:", confirmation)
        if not (is_yes(confirmation) or is_no(confirmation)):
            confirmation = input("Voice confirmation unclear. Type yes or no: ").strip()
    if is_yes(confirmation):
        user_profile["name"] = name
        answer = f"धन्यवाद! मैं आपको {name} कहूँगा।"
    elif is_no(confirmation):
        print("\nRecognized name:", name)
        corrected = input("Enter correct name (or Enter to try again): ").strip()
        if corrected and is_yes(input(f"Save '{corrected}' as your name? (yes/no): ").strip()):
            user_profile["name"] = corrected
            answer = f"धन्यवाद! मैं आपको {corrected} कहूँगा।"
        else:
            answer = "नाम सेव नहीं किया गया। कृपया अपना नाम दोबारा बताइए।"
    else:
        answer = "नाम की पुष्टि नहीं हुई। कृपया दोबारा कोशिश करें।"
    remember_exchange(f"Name confirmation: {confirmation}", answer)
    speak_answer(answer)


def process_question(question):
    name = extract_name(question)
    if name:
        handle_name_confirmation(name)
    else:
        speak_answer(ask_vedamrit(question))


def main():
    print("\n================================")
    print("     VEDAMRIT VOICE ASSISTANT")
    print("================================")
    if TEXT_TEST_MODE:
        print("Text mode. Type exit to quit.")
        while True:
            question = input("\nYou: ").strip()
            if question.lower() == "exit":
                break
            if question:
                try:
                    process_question(question)
                except Exception as exc:
                    print("Error:", exc)
        return
    load_asr()
    print("\nLanguage: Hindi | Speech Model: IndicConformer RNNT")
    print("AI Model:", OLLAMA_MODEL)
    print("Press Enter to speak; type exit to quit")
    while True:
        try:
            command = input("\nPress Enter to speak (exit to quit): ").strip()
            if command.lower() == "exit":
                print("\nVedAmrit stopped.")
                break
            question = recognize_voice(record_voice())
            print("\nYou:", question)
            if question:
                process_question(question)
            else:
                print("No speech detected. Try again.")
        except KeyboardInterrupt:
            print("\nVedAmrit stopped.")
            break
        except Exception as exc:
            print("\nError:", exc)


if __name__ == "__main__":
    main()
