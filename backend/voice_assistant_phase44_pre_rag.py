"""VedAmrit Phase 4.4: speech clarification + natural Hindi conversation with IndicConformer RNNT and VAD.
Run: python backend/voice_assistant_phase43.py
"""
import re
import time
from collections import deque
import unicodedata
import os

import numpy as np
import sounddevice as sd
import torch
from transformers import AutoModel
from ollama import chat
from offline_tts import speak
from context_manager import VedAmritContextManager
from persistent_memory import PersistentMemory
from natural_hindi import conversational_reply, tidy_llm_response
from speech_clarification import SpeechClarifier

SAMPLE_RATE = 16000
# Automatic recording settings
FRAME_MS = 30
START_TIMEOUT = 5.0
MAX_RECORD_SECONDS = 15.0
END_SILENCE_SECONDS = 0.9
MIN_SPEECH_SECONDS = 0.25
START_RMS_THRESHOLD = 0.008
CONTINUE_RMS_THRESHOLD = 0.004
PRE_ROLL_SECONDS = 0.3
ASR_MODEL_ID = "ai4bharat/indic-conformer-600m-multilingual"
OLLAMA_MODEL = "gemma3:4b"
TEXT_TEST_MODE = False
DEBUG_LLM = os.getenv("VEDAMRIT_DEBUG_LLM", "0") == "1"
MAX_CONTEXT_EXCHANGES = 5

user_profile = {"name": None}
SYSTEM_PROMPT = (
    "तुम वेदअमृत नाम के एक दोस्ताना, विनम्र हिंदी वॉइस असिस्टेंट हो। "
    "आम बोलचाल की सरल, साफ और स्वाभाविक हिंदी में जवाब दो। "
    "सामान्य सवाल का जवाब अधिकतर एक छोटे वाक्य में, जरूरत हो तो दो वाक्यों में दो। "
    "पहले सवाल का सीधा जवाब दो; हर जवाब के आखिर में सवाल मत पूछो। "
    "अनावश्यक तारीफ, औपचारिक भूमिका, दोहराव और लंबे भाषण से बचो। "
    "अगर यूज़र सिर्फ हाँ, नहीं या पनीर जैसा छोटा जवाब दे, तो केवल प्रासंगिक पिछले संदर्भ से अर्थ समझो। "
    "संदर्भ नहीं है तो अनुमान लगाने के बजाय एक छोटा स्पष्ट सवाल पूछो। "
    "यूज़र ने जो नहीं बताया उसे उसकी पसंद या निजी तथ्य बनाकर मत कहो। "
    "तुम AI हो: तुम खाना नहीं खाते, बाहर नहीं जाते, संगीत चला नहीं सकते और वास्तविक दुनिया में काम नहीं करते। "
    "आयुर्वेदिक RAG अभी जुड़ा नहीं है; दवा या खुराक न बताओ। "
    "स्वास्थ्य, सुरक्षा या गंभीर परेशानी पर संक्षिप्तता के कारण जरूरी चेतावनी मत हटाओ।"
)

context_manager = VedAmritContextManager(max_exchanges=MAX_CONTEXT_EXCHANGES)
persistent_memory = PersistentMemory()
user_profile["name"] = persistent_memory.get("name")

asr_model = None
clarifier = SpeechClarifier()


def load_asr():
    global asr_model
    if asr_model is None:
        print("Loading IndicConformer RNNT (first run may take time)...")
        asr_model = AutoModel.from_pretrained(ASR_MODEL_ID, trust_remote_code=True)
        asr_model.eval()
        print("IndicConformer ready!")


def record_voice():
    """Record speech with simple RMS-based start/end detection.

    Press Enter in the main loop to begin listening. Returns a mono
    float32 NumPy waveform at 16 kHz, or an empty waveform on timeout.
    """
    block_size = int(SAMPLE_RATE * FRAME_MS / 1000)
    pre_roll_count = max(1, round(PRE_ROLL_SECONDS * 1000 / FRAME_MS))
    pre_roll = deque(maxlen=pre_roll_count)
    captured = []
    started = False
    voiced_frames = 0
    silence_frames = 0
    silence_limit = max(1, round(END_SILENCE_SECONDS * 1000 / FRAME_MS))
    min_voiced_frames = max(1, round(MIN_SPEECH_SECONDS * 1000 / FRAME_MS))

    print("\nListening... Speak naturally.")
    start_time = time.monotonic()

    # Blocking reads keep the audio pipeline simple and avoid writing files.
    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
        blocksize=block_size,
    ) as stream:
        while True:
            frame, overflowed = stream.read(block_size)
            if overflowed:
                print("Warning: microphone buffer overflow detected.")
            frame = frame.reshape(-1).copy()
            rms = float(np.sqrt(np.mean(frame * frame)))
            elapsed = time.monotonic() - start_time

            if not started:
                pre_roll.append(frame)
                if rms >= START_RMS_THRESHOLD:
                    started = True
                    captured.extend(pre_roll)
                    voiced_frames = 1
                    silence_frames = 0
                    print("Speech detected. Recording...")
                elif elapsed >= START_TIMEOUT:
                    print("No speech started within 5 seconds.")
                    return np.array([], dtype=np.float32)
                continue

            captured.append(frame)
            if rms >= CONTINUE_RMS_THRESHOLD:
                voiced_frames += 1
                silence_frames = 0
            else:
                silence_frames += 1

            if elapsed >= MAX_RECORD_SECONDS:
                print("Maximum recording duration reached.")
                break
            if silence_frames >= silence_limit:
                print("Silence detected. Recording stopped.")
                break

    if voiced_frames < min_voiced_frames:
        print("Recording was too short to confirm speech.")
        return np.array([], dtype=np.float32)

    audio = np.concatenate(captured).astype(np.float32)
    total_rms = float(np.sqrt(np.mean(audio * audio)))
    print(f"Recorded: {len(audio) / SAMPLE_RATE:.2f} seconds")
    print(f"Microphone RMS: {total_rms:.5f}")
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
    if re.search(r"(?:कैसेहो|कैसीहो|कैसेहैं)", compact) or compact in {"तुमपैसेहो", "तुमकैसेहो", "आपकैसेहो"}:
        return "how_are_you"
    if any(phrase in compact for phrase in ("तुमक्याकरसकतेहो", "आपक्याकरसकतेहैं", "तुम्हारेकामक्या", "तुमक्याकरतेहो", "तुम्हारीक्षमताक्याहै")):
        return "capabilities"
    if any(phrase in compact for phrase in ("मेरानामबदलो", "मेरानामसुधारो", "मेरानामगलतहै", "नामबदलो", "नामसुधारो")):
        return "correct_name"
    if compact in {"सुधारो", "नामसुधारो", "नामबदलो"}:
        return "correct_name"
    if any(phrase in compact for phrase in ("नमस्ते", "नमस्कार", "हेलो", "हाय")) and len(compact) < 25:
        return "greeting"
    if any(word in compact for word in ("चूतिय", "चुतिय", "चुतीय", "बेवकूफ", "गाली")):
        return "abuse"
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
    """Store the completed exchange in filtered, in-session memory."""
    context_manager.remember(question, answer)
    preference = persistent_memory.extract_preference(question)
    if preference is not None:
        persistent_memory.set("food_preference", preference)



def repeated_or_broken(answer):
    answer = normalize(answer).replace("\ufffd", "")
    if not answer or len(answer) > 400:
        return True
    bad_fragments = ("एक आवश्यक अवस्था", "जीवन के विभिन्न अंग", "बीमारी आती है")
    if any(fragment in answer for fragment in bad_fragments):
        return True
    words = re.findall(r"[\w\u0900-\u097f]+", answer)
    if len(words) >= 9:
        for n in (1, 2, 3):
            chunks = [tuple(words[i:i+n]) for i in range(0, len(words)-n+1, n)]
            if chunks and max(chunks.count(chunk) for chunk in set(chunks)) >= 4:
                return True
    return False


def llm_answer(question):
    # Keep context in natural conversation order. Do not insert a system message
    # between the user's last turn and the current question.
    messages = [{"role": "system", "content": SYSTEM_PROMPT +
                 f" यूज़र का पुष्टि किया हुआ नाम: {user_profile['name'] or 'अज्ञात'}."}]
    messages += context_manager.get_context(question)
    messages.append({"role": "user", "content": question})
    if DEBUG_LLM:
        print("[DEBUG] Selected LLM messages:", messages)
    for attempt in range(2):
        prompt = messages
        if attempt:
            prompt = messages[:-1] + [
                {"role": "system", "content": "पिछला जवाब अस्पष्ट या दोहराव वाला था। सीधे, सरल हिंदी में जवाब दो; आवश्यक स्वास्थ्य चेतावनी न छोड़ो।"},
                messages[-1],
            ]
        response = chat(
            model=OLLAMA_MODEL, messages=prompt, think=False,
            options={"temperature": 0.45 if attempt == 0 else 0.65,
                     "repeat_penalty": 1.18, "repeat_last_n": 96,
                     "num_predict": 110, "num_ctx": 3072},
        )
        answer = tidy_llm_response(normalize(response["message"]["content"]).replace("\ufffd", ""))
        if DEBUG_LLM:
            print(f"[DEBUG] LLM attempt {attempt + 1}: {answer}")
        if not repeated_or_broken(answer):
            return answer
    return "माफ कीजिए, मेरा जवाब ठीक नहीं बन पाया। क्या आप एक बार फिर पूछेंगे?"


def recall_preference(question):
    """Recall explicit preferences from durable storage, then short-term context."""
    if not context_manager.is_preference_question(question):
        return None
    preference = persistent_memory.get("food_preference")
    if preference:
        return f"आपने बताया था कि आपको {preference} पसंद है।"
    return "आपने अभी तक अपनी पसंद स्पष्ट रूप से नहीं बताई है।"


def memory_command(question):
    """User-controlled memory commands; deletion requires confirmation."""
    compact = compact_text(question)
    if compact in {"मेरीयाददाश्तदिखाओ", "मेरीमेमोरीदिखाओ", "क्यायादहै", "तुम्हेंक्यायादहै"}:
        facts = persistent_memory.all()
        if not facts:
            return "अभी मेरी स्थायी याददाश्त खाली है।"
        parts = []
        if facts.get("name"):
            parts.append(f"आपका नाम {facts['name']} है")
        if facts.get("food_preference"):
            parts.append(f"आपको {facts['food_preference']} पसंद है")
        return "मुझे याद है: " + "; ".join(parts) + "।" if parts else "अभी मेरी स्थायी याददाश्त खाली है।"
    if compact in {"मेरीपसंदभूलजाओ", "मेरीपसंदमिटाओ"}:
        if is_yes(input("Delete saved preference? Type yes/no: ").strip()):
            persistent_memory.delete("food_preference")
            return "ठीक है, मैंने सेव की हुई पसंद मिटा दी है।"
        return "ठीक है, पसंद सेव रहेगी।"
    if compact in {"मेरानामभूलजाओ", "मेरानाममिटाओ"}:
        if is_yes(input("Delete saved name? Type yes/no: ").strip()):
            persistent_memory.delete("name")
            user_profile["name"] = None
            return "ठीक है, मैंने सेव किया हुआ नाम मिटा दिया है।"
        return "ठीक है, नाम सेव रहेगा।"
    if compact in {"सबकुछभूलजाओ", "मेरीसारीमेमोरीमिटाओ"}:
        if is_yes(input("Delete ALL saved facts? Type yes/no: ").strip()):
            persistent_memory.clear()
            user_profile["name"] = None
            context_manager.reset()
            return "ठीक है, मैंने सेव की हुई जानकारी मिटा दी है।"
        return "ठीक है, सेव की हुई जानकारी नहीं मिटाई।"
    return None


def ask_vedamrit(question):
    question = normalize(question)
    compact = compact_text(question)
    intent = detect_intent(question)
    command_answer = memory_command(question)
    preference_answer = recall_preference(question) if command_answer is None else None
    if command_answer is not None:
        answer = command_answer
    elif preference_answer is not None:
        answer = preference_answer
    elif intent == "ask_assistant_name":
        answer = "मेरा नाम वेदअमृत है।"
    elif intent == "ask_user_name":
        answer = (f"आपका नाम {user_profile['name']} है।" if user_profile["name"]
                  else "आपने अभी तक अपना नाम नहीं बताया है।")
    elif intent == "how_are_you":
        answer = "मैं ठीक हूँ, धन्यवाद! आप कैसे हैं?"
    elif intent == "greeting":
        answer = f"नमस्ते{(' ' + user_profile['name']) if user_profile['name'] else ''}! बताइए, कैसे मदद करूँ?"
    elif intent == "abuse":
        answer = "मैं आपकी मदद करने के लिए यहाँ हूँ। आप अपना सवाल बता सकते हैं।"
    elif intent == "capabilities":
        answer = ("मैं हिंदी में बातचीत कर सकता हूँ, आपके सवालों के जवाब दे सकता हूँ "
                  "और सामान्य विषय समझा सकता हूँ। आयुर्वेदिक जानकारी की सुविधा अभी विकसित हो रही है।")
    elif compact in {"खानाखाया", "खानाखायाक्या", "तुमनेखानाखाया", "आपनेखानाखाया"}:
        answer = "मैं AI हूँ, इसलिए खाना नहीं खाता। आपने क्या खाया?"
    elif compact in {"मैंठीकहूँ", "मैंठीकहूं", "मैंठीकहूँबो", "मैंठीकहूंवो", "मैंभीठीकहूँ"}:
        answer = "अच्छा लगा सुनकर! बताइए, आज किस बारे में बात करें?"
    else:
        recent = context_manager.exchanges[-1] if context_manager.exchanges else None
        answer = conversational_reply(question, recent)
        if answer is None:
            answer = llm_answer(question)
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
        persistent_memory.set("name", name)
        answer = f"धन्यवाद! मैं आपको {name} कहूँगा।"
    elif is_no(confirmation):
        print("\nRecognized name:", name)
        corrected = input("Enter correct name (or Enter to try again): ").strip()
        if corrected and is_yes(input(f"Save '{corrected}' as your name? (yes/no): ").strip()):
            user_profile["name"] = corrected
            persistent_memory.set("name", corrected)
            answer = f"धन्यवाद! मैं आपको {corrected} कहूँगा।"
        else:
            answer = "नाम सेव नहीं किया गया। कृपया अपना नाम दोबारा बताइए।"
    else:
        answer = "नाम की पुष्टि नहीं हुई। कृपया दोबारा कोशिश करें।"
    remember_exchange(f"Name confirmation: {confirmation}", answer)
    speak_answer(answer)


def process_question(question):
    # Do not send incomplete/unclear speech to the LLM or persistent memory.
    clarification = clarifier.check(question)
    if clarification is not None:
        speak_answer(clarification)
        return
    if detect_intent(question) == "correct_name":
        print("\nCurrent saved name:", user_profile["name"] or "Not set")
        corrected = input("Type your correct name (Enter to cancel): ").strip()
        if not corrected:
            speak_answer("ठीक है, नाम नहीं बदला गया।")
            return
        confirmation = input(f"Save '{corrected}' as your name? (yes/no): ").strip()
        if is_yes(confirmation):
            user_profile["name"] = corrected
            persistent_memory.set("name", corrected)
            speak_answer(f"ठीक है, अब मैं आपको {corrected} कहूँगा।")
        else:
            speak_answer("ठीक है, नाम नहीं बदला गया।")
        return
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
    print("Persistent memory:", persistent_memory.db_path)
    print("Press Enter to speak; type exit to quit")
    print("Say 'मेरा नाम सुधारो' to correct a saved name.")
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
                speak_answer(clarifier.no_speech())
        except KeyboardInterrupt:
            print("\nVedAmrit stopped.")
            break
        except Exception as exc:
            print("\nError:", exc)


if __name__ == "__main__":
    main()
