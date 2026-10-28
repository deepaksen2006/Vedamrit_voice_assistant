from pathlib import Path
import fitz

ROOT = Path(__file__).resolve().parent.parent
BOOKS = ROOT / "books"
OUTPUT = ROOT / "storage" / "extracted_books"
OUTPUT.mkdir(parents=True, exist_ok=True)

for name in ["charaka_samhita", "sushruta_samhita"]:
    pdf_path = BOOKS / f"{name}.pdf"

    if not pdf_path.exists():
        print(f"Missing PDF: {pdf_path}")
        continue

    doc = fitz.open(pdf_path)
    extracted_pages = []
    readable_pages = 0

    for page_number, page in enumerate(doc, start=1):
        text = page.get_text("text").strip()

        if len(text) >= 100:
            readable_pages += 1

        extracted_pages.append(
            f"\n--- PAGE {page_number} ---\n{text}\n"
        )

    output_path = OUTPUT / f"{name}.txt"
    output_path.write_text(
        "\n".join(extracted_pages),
        encoding="utf-8"
    )

    print(f"\nBook: {name}")
    print(f"Total pages: {len(doc)}")
    print(f"Pages with >=100 extracted characters: {readable_pages}")
    print(f"Output: {output_path}")
    print("Note: Character count does not prove text is readable.")

    doc.close()
