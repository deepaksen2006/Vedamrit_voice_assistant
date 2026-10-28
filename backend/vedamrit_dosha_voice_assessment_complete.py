"""VedAmrit bilingual Dosha assessment: spoken questions, microphone answers.

Place beside voice_assistant_phase44.py and vedamrit_dosha_questions_bilingual.json.
Run: python backend/vedamrit_dosha_voice_assessment_complete.py
Does NOT invent dosha scoring; answers are saved for your existing scoring code.
English speaking speed is fixed at 2 (Windows System.Speech Rate).
Hindi speed remains controlled by offline_tts.py.
"""
from __future__ import annotations
import json
import re
import sys
import subprocess
import base64
from pathlib import Path

HERE = Path(__file__).resolve().parent
QUESTIONS = HERE / 'vedamrit_dosha_questions_bilingual.json'

WORDS = {
    'hi': {1: ('एक', 'पहला', 'पहली', '1', '१'), 2: ('दो', 'दूसरा', 'दूसरी', '2', '२'),
           3: ('तीन', 'तीसरा', 'तीसरी', '3', '३'), 4: ('चार', 'चौथा', 'चौथी', '4', '४')},
    'en': {1: ('one', 'first', '1'), 2: ('two', 'second', '2'),
           3: ('three', 'third', '3'), 4: ('four', 'fourth', '4')},
}
YES = {'hi': {'हाँ', 'हां', 'हा', 'जी', 'सही', 'बिल्कुल'}, 'en': {'yes', 'yeah', 'correct', 'right'}}
NO = {'hi': {'नहीं', 'नही', 'ना', 'गलत'}, 'en': {'no', 'incorrect', 'wrong'}}
STOP = {'hi': {'रुको', 'बंद करो', 'बंद', 'स्टॉप'}, 'en': {'stop', 'exit', 'quit'}}
REPEAT = {'hi': {'दोबारा', 'फिर से', 'दोबारा बोलो'}, 'en': {'repeat', 'repeat question'}}


def norm(text):
    return re.sub(r'\s+', ' ', re.sub(r'[^\w\u0900-\u097f]+', ' ', str(text).lower())).strip()


def choice_from_speech(heard, options, lang):
    heard = norm(heard)
    if not heard:
        return None
    # Handle common English speech-to-text forms such as 'option number two'.
    found = set()
    for number, aliases in WORDS[lang].items():
        if number > len(options):
            continue
        for alias in aliases:
            if re.search(r'(?<!\w)' + re.escape(norm(alias)) + r'(?!\w)', heard):
                found.add(number - 1)
    for idx, option in enumerate(options):
        if heard == norm(option):
            found.add(idx)
    return next(iter(found)) if len(found) == 1 else None


class SpeechIO:
    def __init__(self, lang, typed=False, english_rate=2):
        self.lang = lang
        self.typed = typed
        self.english_rate = max(-10, min(10, int(english_rate)))
        self._voice = None
        self._english_tts = None
        self._english_asr = None
        if typed:
            return
        # Use the already-working VedAmrit microphone for BOTH languages.
        import voice_assistant_phase44 as voice
        self._voice = voice
        if lang == 'hi':
            from offline_tts import speak
            self._hindi_speak = speak
        else:
            # Windows System.Speech is synchronous and avoids pyttsx3's
            # repeated runAndWait() issue, which can silently skip options.
            if sys.platform != 'win32':
                raise RuntimeError('English spoken assessment currently requires Windows.')
            print('[TTS] English: Windows System.Speech (synchronous per utterance)', flush=True)
            try:
                from faster_whisper import WhisperModel
            except ImportError as exc:
                raise RuntimeError('Install English speech recognition: python -m pip install faster-whisper') from exc
            print('[ASR] Loading English Whisper model (first run downloads model files)...', flush=True)
            try:
                # base.en is English-specific; CPU/int8 keeps it usable on Windows.
                self._english_asr = WhisperModel('base.en', device='cpu', compute_type='int8')
            except Exception as exc:
                raise RuntimeError('Could not load English Whisper model. Internet is required once to download it; later runs use the cache. Details: ' + str(exc)) from exc
            print('[ASR] English Whisper ready!', flush=True)

    def speak(self, text):
        print('\nVedAmrit:', text, flush=True)
        if self.typed:
            return
        if self.lang == 'hi':
            self._hindi_speak(text)
            return
        # A fresh PowerShell process per utterance is slower but reliable:
        # it blocks until the entire option has finished speaking.
        # Encode the text as Base64 to preserve punctuation and prevent
        # PowerShell injection from questionnaire content.
        payload = base64.b64encode(text.encode('utf-8')).decode('ascii')
        script = (
            'Add-Type -AssemblyName System.Speech; '
            '$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; '
            f'$s.Rate = {self.english_rate}; $s.Volume = 100; '
            '$t = [Text.Encoding]::UTF8.GetString([Convert]::FromBase64String("' + payload + '")); '
            'try { $s.Speak($t) } finally { $s.Dispose() }'
        )
        encoded_script = base64.b64encode(script.encode('utf-16-le')).decode('ascii')
        try:
            proc = subprocess.run(
                ['powershell.exe', '-NoProfile', '-NonInteractive',
                 '-EncodedCommand', encoded_script],
                capture_output=True, text=True, timeout=120,
                creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0),
            )
        except Exception as exc:
            raise RuntimeError(f'English speech playback failed: {exc}') from exc
        if proc.returncode != 0:
            raise RuntimeError('English TTS failed: ' + (proc.stderr or proc.stdout).strip())
        print('[TTS] Finished speaking.', flush=True)

    def listen(self):
        if self.typed:
            return input('You (type answer): ').strip()
        try:
            print('[MIC] Listening now. Speak after this message.', flush=True)
            audio = self._voice.record_voice()
            if audio is None or len(audio) == 0:
                print('[MIC] No audio captured.')
                return ''
            import numpy as np
            waveform = np.asarray(audio, dtype=np.float32).flatten()
            rms = float(np.sqrt(np.mean(waveform ** 2)))
            print(f'[MIC] RMS={rms:.5f}, seconds={len(waveform)/16000:.2f}')
            if rms < 0.003:
                print('[MIC] Audio is too quiet; check Windows microphone input.')
                return ''
            if self.lang == 'hi':
                heard = str(self._voice.recognize_voice(waveform)).strip()
            else:
                segments, info = self._english_asr.transcribe(
                    waveform, language='en', beam_size=3,
                    vad_filter=False, condition_on_previous_text=False,
                    initial_prompt='Option one, option two, option three, option four. Yes. No. Repeat. Stop.'
                )
                heard = ' '.join(segment.text.strip() for segment in segments).strip()
            print('[ASR] Heard:', repr(heard))
            return heard
        except Exception as exc:
            print('[MIC/ASR ERROR]', repr(exc))
            return ''


def save_answers(path, language, answers):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps({'language': language, 'answers': answers}, ensure_ascii=False, indent=2), encoding='utf-8')
    temp.replace(path)


def load_answers(path, language):
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding='utf-8'))
    return data.get('answers', {}) if data.get('language') == language else {}


def run_assessment(language, typed=False, english_rate=2):
    if not QUESTIONS.exists():
        raise FileNotFoundError(f'Missing questionnaire: {QUESTIONS}')
    data = json.loads(QUESTIONS.read_text(encoding='utf-8'))
    questions = data['languages'][language]
    io = SpeechIO(language, typed=typed, english_rate=english_rate)
    path = HERE / f'dosha_voice_answers_{language}.json'
    answers = load_answers(path, language)
    number_words = {1: 'एक', 2: 'दो', 3: 'तीन', 4: 'चार'}

    def say_question(q, idx):
        io.speak((f'प्रश्न {idx + 1}। ' if language == 'hi' else f'Question {idx + 1}. ') + q['question'])
        for n, option in enumerate(q['options'], 1):
            label = f'विकल्प {number_words[n]}। ' if language == 'hi' else f'Option {n}. '
            io.speak(label + option)

    for idx, q in enumerate(questions):
        key = f"{q['section']}:{q['id']}"
        if key in answers:
            continue
        say_question(q, idx)
        failed = 0
        while True:
            io.speak('अपना विकल्प बोलिए।' if language == 'hi' else 'Please say your option number.')
            heard = norm(io.listen())
            print(f'[VOICE DOSHA] Q={key} heard={heard!r}')
            if heard in STOP[language]:
                save_answers(path, language, answers)
                print('[PAUSED] Saved answers:', path)
                return answers
            if heard in REPEAT[language]:
                say_question(q, idx)
                continue
            choice = choice_from_speech(heard, q['options'], language)
            if choice is None:
                failed += 1
                if failed >= 3:
                    print('[FALLBACK] Three unclear microphone answers.')
                    fallback = input(f'Type option number 1-{len(q["options"])} (or q to pause): ').strip()
                    if fallback.lower() == 'q':
                        save_answers(path, language, answers)
                        return answers
                    choice = int(fallback) - 1 if fallback.isdigit() and 1 <= int(fallback) <= len(q['options']) else None
                    failed = 0
                if choice is None:
                    io.speak('उत्तर स्पष्ट नहीं है। दोबारा प्रयास करें।' if language == 'hi' else 'I could not understand. Please try again.')
                    continue
            label = number_words[choice + 1] if language == 'hi' else str(choice + 1)
            io.speak(f'आपने विकल्प {label} चुना है। क्या यह सही है? हाँ या नहीं बोलिए।' if language == 'hi'
                     else f'You selected option {label}. Is that correct? Say yes or no.')
            confirmation = norm(io.listen())
            if confirmation in NO[language]:
                continue
            if confirmation not in YES[language]:
                print('[CONFIRMATION] Not recognized; please confirm by keyboard.')
                confirmation = norm(input('Confirm selection? (yes/no): '))
                if confirmation not in YES['en'] and confirmation not in YES['hi']:
                    continue
            answers[key] = {'option_index': choice, 'option_number': choice + 1,
                            'option_text': q['options'][choice]}
            save_answers(path, language, answers)
            print(f'[SAVED] {key}: option {choice + 1} ({len(answers)}/{len(questions)})')
            break
    io.speak('आकलन पूरा हुआ।' if language == 'hi' else 'Assessment complete.')
    print('[RESULT] Answers saved:', path)
    print('[NOTE] Apply your existing text assessment scoring logic to these answers.')
    return answers


if __name__ == '__main__':
    try:
        lang = input('Language (hi/en): ').strip().lower()
        if lang not in ('hi', 'en'):
            raise ValueError('Choose hi or en')
        mode = input('Mode (voice/text): ').strip().lower()
        if mode not in ('voice', 'text'):
            raise ValueError('Choose voice or text')
        english_rate = 2  # Fixed English speaking speed; no speed prompt.
        run_assessment(lang, typed=(mode == 'text'), english_rate=english_rate)
    except KeyboardInterrupt:
        print('\nStopped. Previously confirmed answers remain saved.')
    except Exception as exc:
        print('[ERROR]', exc)
        sys.exit(1)
