"""Bilingual, resumable Dosha questionnaire voice controller.

This module does not calculate Dosha scores: pass recorded option indexes to the
existing text-assessment scoring implementation. Hindi/English TTS and ASR are
provided by the application as callbacks.
"""
from __future__ import annotations
import json
import re
from pathlib import Path
from offline_tts import speak
def speak_assessment(text, language="hi"):
    """Speak assessment questions aloud."""
    print("VedAmrit:", text)

    if language == "hi":
        try:
            speak(text)
        except Exception as exc:
            print("[TTS ERROR]", exc)
    else:
        print("[TTS] English voice is not configured yet.")
HERE = Path(__file__).resolve().parent
QUESTION_FILE = HERE / 'vedamrit_dosha_questions_bilingual.json'

NUMBERS = {
    'hi': {
        1: ('1','एक','पहला','पहली','विकल्प एक','विकल्प 1'),
        2: ('2','दो','दूसरा','दूसरी','विकल्प दो','विकल्प 2'),
        3: ('3','तीन','तीसरा','तीसरी','विकल्प तीन','विकल्प 3'),
        4: ('4','चार','चौथा','चौथी','विकल्प चार','विकल्प 4'),
    },
    'en': {
        1: ('1','one','first','option one','option 1'),
        2: ('2','two','second','option two','option 2'),
        3: ('3','three','third','option three','option 3'),
        4: ('4','four','fourth','option four','option 4'),
    },
}

def normalize(text):
    return re.sub(r'\s+', ' ', re.sub(r'[^\w\u0900-\u097f]+', ' ', str(text).lower())).strip()

def interpret_answer(utterance, options, language):
    """Return zero-based option index or None. Never guess ambiguous speech."""
    heard = normalize(utterance)
    if not heard:
        return None
    matches = set()
    for number, phrases in NUMBERS[language].items():
        if number > len(options):
            continue
        for phrase in phrases:
            if re.search(r'(?<!\w)' + re.escape(normalize(phrase)) + r'(?!\w)', heard):
                matches.add(number - 1)
    for i, option in enumerate(options):
        if heard == normalize(option):
            matches.add(i)
    return next(iter(matches)) if len(matches) == 1 else None

class VoiceDoshaAssessment:
    def __init__(self, language, speak, listen, save_path=None, question_file=QUESTION_FILE):
        if language not in ('hi','en'):
            raise ValueError('language must be hi or en')
        self.language = language
        self.speak = speak  # callable(text: str), must support selected language
        self.listen = listen  # callable() -> recognized text in selected language
        self.save_path = Path(save_path or HERE / f'dosha_voice_answers_{language}.json')
        self.questions = json.loads(Path(question_file).read_text(encoding='utf8'))['languages'][language]
        self.answers = {}
        if self.save_path.exists():
            previous = json.loads(self.save_path.read_text(encoding='utf8'))
            if previous.get('language') == language:
                self.answers = previous.get('answers', {})

    def save(self):
        payload = {'language':self.language, 'answers':self.answers}
        temp = self.save_path.with_suffix(self.save_path.suffix + '.tmp')
        temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf8')
        temp.replace(self.save_path)

    def run(self):
        i = 0
        while i < len(self.questions):
            q = self.questions[i]
            key = f"{q['section']}:{q['id']}"
            if key in self.answers:
                i += 1
                continue
            self.speak((f"प्रश्न {i+1}। " if self.language=='hi' else f"Question {i+1}. ") + q['question'])
            for n, option in enumerate(q['options'],1):
                self.speak((f'विकल्प {n}: ' if self.language=='hi' else f'Option {n}: ') + option)
            while True:
                self.speak('अपना विकल्प बोलिए।' if self.language=='hi' else 'Please say your option number.')
                heard = str(self.listen() or '').strip()
                print(f'[VOICE DOSHA] Q={key} heard={heard!r}')
                normalized = normalize(heard)
                if normalized in ('रुको','बंद करो','stop','exit','quit'):
                    self.save()
                    return self.answers
                if normalized in ('दोबारा','फिर से','repeat','repeat question'):
                    self.speak(q['question'])
                    for n, option in enumerate(q['options'],1):
                        self.speak((f'विकल्प {n}: ' if self.language=='hi' else f'Option {n}: ') + option)
                    continue
                choice = interpret_answer(heard, q['options'], self.language)
                if choice is None:
                    self.speak('उत्तर स्पष्ट नहीं है। कृपया विकल्प संख्या दोबारा बोलिए।' if self.language=='hi' else 'I did not understand. Please repeat the option number.')
                    continue
                self.speak((f'आपने विकल्प {choice+1} चुना। सही है? हाँ या नहीं बोलिए।' if self.language=='hi' else f'You selected option {choice+1}. Is that correct? Say yes or no.'))
                confirm = normalize(self.listen() or '')
                if confirm in ('हाँ','हां','हा','जी','सही','yes','yeah','correct'):
                    self.answers[key] = {'option_index':choice, 'option_number':choice+1, 'option_text':q['options'][choice]}
                    self.save()
                    i += 1
                    break
                if confirm in ('नहीं','नही','ना','no','incorrect'):
                    continue
                self.speak('पुष्टि नहीं हुई। फिर से विकल्प बताइए।' if self.language=='hi' else 'Not confirmed. Please select an option again.')
        self.speak('आकलन पूरा हुआ।' if self.language=='hi' else 'Assessment complete.')
        return self.answers

if __name__ == '__main__':
    # Safe text-only demonstration. Replace these callbacks with actual ASR/TTS.
    language = input('Language (hi/en): ').strip().lower()
    engine = VoiceDoshaAssessment(language, speak=lambda text: print('VedAmrit:', text), listen=lambda: input('You: '))
    engine.run()
