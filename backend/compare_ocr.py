
from pathlib import Path
from io import BytesIO
import fitz
import pytesseract
from PIL import Image, ImageOps, ImageEnhance

ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "storage" / "ocr_comparison"
OUTPUT.mkdir(parents=True, exist_ok=True)

pytesseract.pytesseract.tesseract_cmd = str(
    Path.home() / "AppData/Local/Programs/Tesseract-OCR/tesseract.exe"
)

SAMPLES = {
    "charaka": ("charaka_samhita.pdf", 50),
    "sushruta": ("sushruta_samhita.pdf", 40),
}

def render_page(pdf_path, page_number):
    with fitz.open(pdf_path) as doc:
        page = doc[page_number - 1]
        pix = page.get_pixmap(
            matrix=fitz.Matrix(300 / 72, 300 / 72),
            colorspace=fitz.csRGB,
            alpha=False,
        )
        return Image.open(BytesIO(pix.tobytes("png"))).convert("RGB")

def preprocess(image):
    gray = ImageOps.grayscale(image)
    gray = ImageOps.autocontrast(gray)
    gray = ImageEnhance.Contrast(gray).enhance(1.3)
    return gray

for book, (filename, page_number) in SAMPLES.items():
    pdf_path = ROOT / "books" / filename
    image = render_page(pdf_path, page_number)
    processed = preprocess(image)

    configs = {
        "baseline": (image, "hin+san", 3),
        "preprocessed": (processed, "hin+san", 3),
        "alternative": (processed, "san+hin", 6),
    }

    for name, (img, language, psm) in configs.items():
        print(f"Processing {book}, page {page_number}: {name}")

        text = pytesseract.image_to_string(
            img,
            lang=language,
            config=f"--oem 1 --psm {psm}",
            timeout=180,
        )

        path = OUTPUT / f"{book}_{name}.txt"
        path.write_text(text, encoding="utf-8")

        print(f"Characters: {len(text)}")
        print(f"Preview: {text[:200]!r}\n")

    processed.save(OUTPUT / f"{book}_processed.png")

print("OCR comparison complete.")
