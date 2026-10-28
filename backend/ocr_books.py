
from pathlib import Path
from io import BytesIO
import argparse

import fitz
import pytesseract
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
BOOKS_DIR = ROOT / "books"
OUTPUT_DIR = ROOT / "storage" / "ocr_books"

TESSERACT_EXE = (
    Path.home()
    / "AppData"
    / "Local"
    / "Programs"
    / "Tesseract-OCR"
    / "tesseract.exe"
)

pytesseract.pytesseract.tesseract_cmd = str(TESSERACT_EXE)

BOOKS = {
    "charaka": "charaka_samhita.pdf",
    "sushruta": "sushruta_samhita.pdf",
}

# Sample pages spread across each PDF.
# Page numbers are PDF pages, not printed book pages.
SAMPLE_PAGES = {
    "charaka": [20, 50, 100, 200, 300],
    "sushruta": [15, 40, 80, 120, 180],
}


def extract_page(page, dpi=220):
    """Render PDF page and extract Devanagari text."""

    zoom = dpi / 72
    pix = page.get_pixmap(
        matrix=fitz.Matrix(zoom, zoom),
        colorspace=fitz.csRGB,
        alpha=False,
    )

    image = Image.open(BytesIO(pix.tobytes("png")))

    text = pytesseract.image_to_string(
        image,
        lang="hin+san",
        config="--oem 1 --psm 3",
        timeout=120,
    )

    return text.strip()


def process_book(book_key, pages=None):
    pdf_path = BOOKS_DIR / BOOKS[book_key]

    if not pdf_path.exists():
        print(f"PDF not found: {pdf_path}")
        return

    output_folder = OUTPUT_DIR / book_key
    output_folder.mkdir(parents=True, exist_ok=True)

    with fitz.open(pdf_path) as document:
        total_pages = len(document)

        selected_pages = (
            pages if pages is not None
            else SAMPLE_PAGES[book_key]
        )

        print(f"\nBook: {book_key}")
        print(f"Total PDF pages: {total_pages}")

        for page_number in selected_pages:
            if not 1 <= page_number <= total_pages:
                print(f"Skipping invalid page {page_number}")
                continue

            print(f"\nProcessing PDF page {page_number}...")

            page = document[page_number - 1]

            try:
                text = extract_page(page)
            except Exception as exc:
                print(f"OCR failed: {exc}")
                continue

            output_path = (
                output_folder
                / f"page_{page_number:04d}.txt"
            )

            output_path.write_text(
                f"BOOK: {book_key}\n"
                f"PDF_PAGE: {page_number}\n"
                f"OCR_LANGUAGE: hin+san\n"
                f"VERIFIED: false\n\n"
                f"{text}",
                encoding="utf-8",
            )

            devanagari_count = sum(
                "\u0900" <= char <= "\u097F"
                for char in text
            )

            print(f"Characters: {len(text)}")
            print(f"Devanagari characters: {devanagari_count}")
            print(f"Saved: {output_path}")
            print(f"Preview: {text[:250]!r}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--book",
        choices=["charaka", "sushruta", "both"],
        default="both",
    )
    parser.add_argument(
        "--pages",
        nargs="+",
        type=int,
        default=None,
        help="Optional 1-based PDF page numbers",
    )

    args = parser.parse_args()

    if not TESSERACT_EXE.is_file():
        raise FileNotFoundError(
            f"Tesseract executable not found: {TESSERACT_EXE}"
        )

    selected_books = (
        list(BOOKS)
        if args.book == "both"
        else [args.book]
    )

    for book in selected_books:
        process_book(book, args.pages)


if __name__ == "__main__":
    main()
