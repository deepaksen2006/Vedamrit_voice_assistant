"""Phase 4.3: short, context-sensitive Hindi conversation helpers (stdlib only)."""
import re

PREFERENCE_RE = re.compile(r"^(?:मुझे|मेरे को)\s+(.{1,65}?)\s+पसंद\s+है[।.!?\s]*$")
FOLLOWUP_QUESTIONS = ("क्या पसंद", "कौन सा", "कौन-सा", "कैसी", "कैसा", "क्यों", "और बताओ")


def conversational_reply(question, last_exchange=None):
    """Return a safe deterministic response for simple social statements; else None."""
    q = question.strip()
    compact = re.sub(r"[\s।.!?,]+", "", q)
    match = PREFERENCE_RE.fullmatch(q)
    if match and not any(x in match.group(1) for x in ("नहीं", "नही", "क्या", "कौन")):
        thing = match.group(1).strip()
        return f"अच्छा! आपको {thing} पसंद है।"  # No forced follow-up.
    if compact in ("धन्यवाद", "शुक्रिया", "थैंक्यू", "बहुतधन्यवाद"):
        return "आपका स्वागत है!"
    if compact in ("ठीकहै", "अच्छा", "समझगया", "समझगई"):
        return "जी, बिल्कुल।"
    if compact in ("हाँ", "हां", "जी", "बिल्कुल") and last_exchange:
        return None  # Let the LLM interpret confirmation with relevant context.
    return None


def tidy_llm_response(text):
    """Remove formatting artifacts without deleting meaningful medical advice."""
    text = re.sub(r"\*\*|__|`", "", text).strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"^(?:वेदअमृत|सहायक|Assistant)\s*[:：-]\s*", "", text, flags=re.I)
    return text
