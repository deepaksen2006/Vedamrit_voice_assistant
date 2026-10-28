
from pathlib import Path
import sqlite3
import re

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "storage" / "ayurveda_rag.db"


def classify_page(text):
    sample = text[:500].lower()

    contents_patterns = [
        r"विषय\s*सूची",
        r"अनुक्रमणिका",
        r"विषयानुक्रम",
        r"table of contents",
        r"contents",
    ]

    for pattern in contents_patterns:
        if re.search(pattern, sample):
            return "CONTENTS_CANDIDATE"

    if len(text.strip()) < 150:
        return "SHORT_CANDIDATE"

    return "CHAPTER_CANDIDATE"


with sqlite3.connect(DB_PATH) as conn:
    columns = [
        row[1]
        for row in conn.execute("PRAGMA table_info(rag_chunks)")
    ]

    if "page_type" not in columns:
        conn.execute(
            "ALTER TABLE rag_chunks "
            "ADD COLUMN page_type TEXT DEFAULT 'UNCLASSIFIED'"
        )

    rows = conn.execute(
        "SELECT id, text FROM rag_chunks"
    ).fetchall()

    counts = {}

    for chunk_id, text in rows:
        page_type = classify_page(text)

        conn.execute(
            "UPDATE rag_chunks SET page_type = ? WHERE id = ?",
            (page_type, chunk_id),
        )

        counts[page_type] = counts.get(page_type, 0) + 1

    conn.commit()

print("RAG CLASSIFICATION COMPLETE")
print("-" * 40)

for category, count in sorted(counts.items()):
    print(f"{category}: {count}")

print(f"\nDatabase updated: {DB_PATH}")
