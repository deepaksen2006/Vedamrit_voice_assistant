"""Conservative Hindi speech clarification for VedAmrit Phase 4.4.

Never guesses what an unclear transcript meant and never saves incomplete speech.
Works on recognized text, not ASR confidence (which this pipeline does not expose).
"""
import re
import unicodedata


class SpeechClarifier:
    def __init__(self):
        self.awaiting_retry = False

    def no_speech(self):
        self.awaiting_retry = True
        return "माफ़ कीजिए, आपकी आवाज़ सुनाई नहीं दी। कृपया दोबारा बोलिए।"

    def check(self, transcript):
        text = unicodedata.normalize("NFKC", str(transcript or "")).strip()
        text = re.sub(r"\\s+", " ", text)
        plain = text.strip(" ।.!?,;:")
        if not plain:
            return self.no_speech()

        # A single filler or abandoned utterance, not a meaningful command.
        if plain in {"अ", "आ", "उम", "हम्म", "हूँ", "मतलब", "वो", "वह", "कुछ"}:
            self.awaiting_retry = True
            return "मैं ठीक से समझ नहीं पाया। कृपया अपना सवाल फिर से बोलिए।"

        # Only very explicit unfinished endings; avoid rejecting short valid questions.
        if re.search(r"(?:\\s|^)(?:के बारे में|इसके बारे में|उसके बारे में|मैं चाहता हूँ कि|मुझे लगता है कि|मुझे कल|कल मुझे|अगर|क्योंकि)$", plain):
            self.awaiting_retry = True
            return "आपकी बात अधूरी लग रही है। कृपया पूरा बताइए।"

        # Corrections without a clear replacement should ask, not hallucinate.
        if plain in {"नहीं मैंने वो नहीं कहा", "मैंने ऐसा नहीं कहा", "गलत सुना", "आपने गलत सुना", "नहीं गलत"}:
            self.awaiting_retry = True
            return "माफ़ कीजिए, मैंने गलत सुना। आप क्या कहना चाहते थे?"

        # If user explicitly corrects the ASR with the actual words, handle them as
        # normal conversation, without altering previously stored facts automatically.
        self.awaiting_retry = False
        return None
