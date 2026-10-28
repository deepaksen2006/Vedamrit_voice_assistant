
from pathlib import Path
import csv
import pymupdf

ROOT = Path(__file__).resolve().parent.parent
BOOKS = ROOT / "books"
AUDIT = ROOT / "storage/ocr_audit/ocr_quality.csv"
OUTPUT = ROOT / "storage/ocr_audit/flagged_images"

FILES = {
    "charaka": "charaka_samhita.pdf",
    "sushruta": "sushruta_samhita.pdf",
}

OUTPUT.mkdir(parents=True, exist_ok=True)

with AUDIT.open(encoding="utf-8-sig", newline="") as f:
    flagged = [
        row for row in csv.DictReader(f)
        if row["status"] != "REVIEW_REQUIRED"
    ]

for book, filename in FILES.items():
    pdf_path = BOOKS / filename

    with pymupdf.open(pdf_path) as doc:
        for row in flagged:
            if row["book"] != book:
                continue

            page_number = int(row["pdf_page"])
            page = doc[page_number - 1]

            pix = page.get_pixmap(
                matrix=pymupdf.Matrix(2, 2),
                alpha=False,
            )

            output_file = (
                OUTPUT / f"{book}_page_{page_number:04d}.png"
            )
            pix.save(output_file)

            print(
                f"{book} page {page_number}: "
                f"{row['status']} -> {output_file.name}"
            )

print("\nInspection images saved to:", OUTPUT)
