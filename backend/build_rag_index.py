
from pathlib import Path
import sqlite3
import json
import re

ROOT = Path(__file__).resolve().parent.parent
SOURCES = ROOT / "storage" / "rag_sources"
DB_PATH = ROOT / "storage" / "ayurveda_rag.db"

DB_PATH.parent.mkdir(parents=True, exist_ok=True)


def make_chunks(text, max_chars=850, overlap=120):
    text = re.sub(r"\s+", " ", text).strip()
    chunks = []

    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))

        if end < len(text):
            boundary = text.rfind(" ", start + max_chars // 2, end)
            if boundary > start:
                end = boundary

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break

        start = max(start + 1, end - overlap)

    return chunks


conn = sqlite3.connect(DB_PATH)

conn.execute("DROP TABLE IF EXISTS rag_fts")
conn.execute("DROP TABLE IF EXISTS rag_chunks")

conn.execute("""
CREATE TABLE rag_chunks (
    id INTEGER PRIMARY KEY,
    book_id TEXT NOT NULL,
    book_title TEXT NOT NULL,
    pdf_page INTEGER NOT NULL,
    chunk_index INTEGER NOT NULL,
    text TEXT NOT NULL,
    verified INTEGER NOT NULL DEFAULT 0
)
""")

conn.execute("""
CREATE VIRTUAL TABLE rag_fts USING fts5(
    text,
    content='rag_chunks',
    content_rowid='id',
    tokenize='unicode61'
)
""")

count = 0

for file in sorted(SOURCES.glob("*.jsonl")):
    with file.open(encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)

            for index, chunk in enumerate(
                make_chunks(record["text"])
            ):
                cursor = conn.execute("""
                    INSERT INTO rag_chunks (
                        book_id, book_title, pdf_page,
                        chunk_index, text, verified
                    )
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    record["book_id"],
                    record["book_title"],
                    record["pdf_page"],
                    index,
                    chunk,
                    int(record.get("verified", False)),
                ))

                conn.execute(
                    "INSERT INTO rag_fts(rowid, text) VALUES (?, ?)",
                    (cursor.lastrowid, chunk),
                )

                count += 1

conn.commit()
conn.close()

print(f"Indexed {count} chunks")
print(f"Database: {DB_PATH}")
print("All chunks remain UNVERIFIED.")
