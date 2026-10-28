
from pathlib import Path
import sqlite3
import argparse
import json
import urllib.request
import urllib.error

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "storage" / "ayurveda_rag.db"
OUTPUT_DIR = ROOT / "storage" / "ai_ocr_suggestions"

OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
MODEL = "qwen2.5:3b"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_chunk(chunk_id):
    with sqlite3.connect(DB_PATH) as conn:
        row = conn.execute("""
            SELECT
                c.book_title,
                c.pdf_page,
                c.text,
                x.corrected_text
            FROM rag_chunks c
            LEFT JOIN rag_corrections x
                ON x.chunk_id = c.id
            WHERE c.id = ?
        """, (chunk_id,)).fetchone()

    if not row:
        raise ValueError(f"Chunk {chunk_id} not found")

    return row


def suggest_correction(text):
    prompt = f"""
You are assisting with OCR correction of historical
Hindi and Sanskrit Ayurvedic literature.

Correct only obvious OCR character and spelling errors.

STRICT RULES:
1. Preserve the original meaning and wording.
2. Do not summarize, translate, or explain.
3. Preserve Sanskrit verses and verse numbers.
4. Preserve Hindi text and punctuation where possible.
5. Do not invent missing words or verses.
6. Do not add medical recommendations.
7. If uncertain, retain the original text.
8. Return ONLY the corrected passage.

OCR TEXT:
{text}
"""

    payload = {
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0
        }
    }

    request = urllib.request.Request(
        OLLAMA_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST"
    )

    with urllib.request.urlopen(
        request, timeout=180
    ) as response:
        result = json.loads(
            response.read().decode("utf-8")
        )

    return result["response"].strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunk", type=int, required=True)
    args = parser.parse_args()

    title, page, original, corrected = load_chunk(
        args.chunk
    )

    # Use the existing manual correction when available.
    source_text = corrected if corrected else original

    print(f"Book: {title}")
    print(f"PDF page: {page}")
    print(f"Chunk: {args.chunk}")
    print(f"Model: {MODEL}")
    print("\nGenerating OCR correction suggestion...")

    try:
        suggestion = suggest_correction(source_text)

    except urllib.error.URLError as error:
        print(f"Ollama connection error: {error}")
        return

    except Exception as error:
        print(f"Error: {error}")
        return

    if not suggestion:
        print("Model returned an empty response.")
        return

    output_file = (
        OUTPUT_DIR / f"chunk_{args.chunk}_suggestion.txt"
    )

    output_file.write_text(
        suggestion,
        encoding="utf-8"
    )

    print("\nAI suggestion saved:")
    print(output_file)
    print("\nThis is NOT verified.")
    print("Compare every word with the original PDF.")


if __name__ == "__main__":
    main()
