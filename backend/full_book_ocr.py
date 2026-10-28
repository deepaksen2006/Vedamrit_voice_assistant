
from pathlib import Path
from io import BytesIO
import argparse
import json

import pymupdf
import pytesseract
from PIL import Image, ImageOps, ImageEnhance

ROOT = Path(__file__).resolve().parent.parent
BOOKS = ROOT / "books"
OUTPUT = ROOT / "storage" / "ocr_full"

pytesseract.pytesseract.tesseract_cmd = str(
    Path.home()
    / "AppData/Local/Programs/Tesseract-OCR/tesseract.exe"
)

FILES = {
    "charaka": "charaka_samhita.pdf",
    "sushruta": "sushruta_samhita.pdf",
}


def ocr_page(page, dpi=300):
    pix = page.get_pixmap(
        matrix=pymupdf.Matrix(dpi / 72, dpi / 72),
        colorspace=pymupdf.csRGB,
        alpha=False,
    )

    image = Image.open(
        BytesIO(pix.tobytes("png"))
    ).convert("RGB")

    image = ImageOps.grayscale(image)
    image = ImageOps.autocontrast(image)
    image = ImageEnhance.Contrast(image).enhance(1.3)

    return pytesseract.image_to_string(
        image,
        lang="hin+san",
        config="--oem 1 --psm 3",
        timeout=180,
    ).strip()


def process_book(book, limit=None):
    pdf_path = BOOKS / FILES[book]
    output_dir = OUTPUT / book
    output_dir.mkdir(parents=True, exist_ok=True)

    if not pdf_path.exists():
        raise FileNotFoundError(pdf_path)

    with pymupdf.open(pdf_path) as doc:
        total = len(doc)
        end = min(total, limit) if limit else total

        print(f"\nProcessing {book}: {end}/{total} pages")

        for index in range(end):
            page_number = index + 1
            output_file = (
                output_dir / f"page_{page_number:04d}.json"
            )

            # Resume after interruption.
            if output_file.exists():
                print(f"Skipping page {page_number}")
                continue

            try:
                text = ocr_page(doc[index])
            except Exception as exc:
                print(f"Page {page_number} failed: {exc}")
                continue

            record = {
                "book": book,
                "pdf_page": page_number,
                "printed_page": None,
                "sthana": None,
                "adhyaya": None,
                "shloka": None,
                "ocr_text": text,
                "ocr_method": "tesseract_hin_san_psm3_300dpi",
                "verified": False,
            }

            output_file.write_text(
                json.dumps(record, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

            print(
                f"Page {page_number}/{end}: "
                f"{len(text)} characters"
            )

    print(f"\nCompleted {book}. Output: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--book",
        choices=["charaka", "sushruta", "both"],
        default="both",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
    )

    args = parser.parse_args()

    selected = (
        list(FILES)
        if args.book == "both"
        else [args.book]
    )

    for book in selected:
        process_book(book, args.limit)
