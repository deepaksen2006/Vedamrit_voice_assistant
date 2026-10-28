
from pathlib import Path
import json
import csv

ROOT = Path(__file__).resolve().parent.parent
OCR_DIR = ROOT / "storage" / "ocr_full"
REPORT_DIR = ROOT / "storage" / "ocr_audit"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

records = []

for book in ("charaka", "sushruta"):
    folder = OCR_DIR / book

    for file in sorted(folder.glob("page_*.json")):
        try:
            data = json.loads(file.read_text(encoding="utf-8"))
            text = data.get("ocr_text", "")

            length = len(text)
            devanagari = sum(
                "\u0900" <= c <= "\u097F" for c in text
            )
            ratio = devanagari / length if length else 0

            if length == 0:
                status = "EMPTY"
            elif length < 300:
                status = "SHORT"
            elif ratio < 0.45:
                status = "LOW_DEVANAGARI"
            else:
                status = "REVIEW_REQUIRED"

            records.append({
                "book": book,
                "pdf_page": data.get("pdf_page"),
                "characters": length,
                "devanagari_ratio": round(ratio, 3),
                "status": status,
                "file": str(file.relative_to(ROOT)),
            })

        except Exception as exc:
            print(f"Failed to read {file}: {exc}")

report_file = REPORT_DIR / "ocr_quality.csv"

with report_file.open(
    "w", newline="", encoding="utf-8-sig"
) as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "book",
            "pdf_page",
            "characters",
            "devanagari_ratio",
            "status",
            "file",
        ],
    )
    writer.writeheader()
    writer.writerows(records)

print("\nOCR QUALITY REPORT")
print("-" * 45)

for book in ("charaka", "sushruta"):
    subset = [r for r in records if r["book"] == book]

    print(f"\n{book.upper()}")
    print(f"Pages checked: {len(subset)}")

    for status in (
        "EMPTY",
        "SHORT",
        "LOW_DEVANAGARI",
        "REVIEW_REQUIRED",
    ):
        count = sum(r["status"] == status for r in subset)
        print(f"{status}: {count}")

print(f"\nReport saved: {report_file}")
