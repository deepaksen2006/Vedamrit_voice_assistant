
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parent.parent
OCR_DIR = ROOT / "storage" / "ocr_full"
OUTPUT_DIR = ROOT / "storage" / "rag_sources"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

EXCLUDE = {
    "charaka": {1, 2, 4, 637, 639},
    "sushruta": {1, 254},
}

BOOK_NAMES = {
    "charaka": "Charaka Samhita",
    "sushruta": "Sushruta Samhita - Sharira Sthana",
}

counts = {}

for book in BOOK_NAMES:
    output_file = OUTPUT_DIR / f"{book}.jsonl"
    kept = 0
    excluded = 0

    with output_file.open("w", encoding="utf-8") as out:
        for file in sorted((OCR_DIR / book).glob("page_*.json")):
            data = json.loads(file.read_text(encoding="utf-8"))
            page = int(data["pdf_page"])

            if page in EXCLUDE[book]:
                excluded += 1
                continue

            text = data.get("ocr_text", "").strip()
            if not text:
                excluded += 1
                continue

            record = {
                "book_id": book,
                "book_title": BOOK_NAMES[book],
                "pdf_page": page,
                "text": text,
                "verified": False,
                "source_file": str(file.relative_to(ROOT)),
            }

            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            kept += 1

    counts[book] = kept
    print(f"{book}: {kept} pages kept, {excluded} excluded")
    print(f"Saved: {output_file}")

print(f"\nTotal RAG source pages: {sum(counts.values())}")
