
from pathlib import Path
import sqlite3
import argparse
import os
import hashlib

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "storage" / "ayurveda_rag.db"

BOOK_FILES = {
    "charaka": ROOT / "books" / "charaka_samhita.pdf",
    "sushruta": ROOT / "books" / "sushruta_samhita.pdf",
}


def text_hash(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def setup_database(conn):
    conn.execute("""
        CREATE TABLE IF NOT EXISTS rag_reviews (
            chunk_id INTEGER PRIMARY KEY,
            decision TEXT NOT NULL
                CHECK(decision IN ('APPROVED', 'REJECTED')),
            reviewed_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(chunk_id) REFERENCES rag_chunks(id)
        )
    """)

    columns = {
        row[1] for row in conn.execute(
            "PRAGMA table_info(rag_reviews)"
        )
    }

    if "reviewed_hash" not in columns:
        conn.execute(
            "ALTER TABLE rag_reviews ADD COLUMN reviewed_hash TEXT"
        )


def review_chunk(conn, chunk_id):
    row = conn.execute("""
        SELECT
            c.book_id,
            c.book_title,
            c.pdf_page,
            c.text,
            x.corrected_text,
            x.status
        FROM rag_chunks c
        LEFT JOIN rag_corrections x ON x.chunk_id = c.id
        WHERE c.id = ?
    """, (chunk_id,)).fetchone()

    if not row:
        print("Chunk not found.")
        return

    book_id, title, page, original, corrected, status = row

    if corrected is None:
        print("No correction found for this chunk.")
        print("Export, edit, and import it first.")
        return

    print("\n" + "=" * 60)
    print(f"Chunk ID: {chunk_id}")
    print(f"Book: {title}")
    print(f"PDF page: {page}")
    print(f"Correction status: {status}")
    print("=" * 60)
    print(corrected)
    print("=" * 60)

    pdf = BOOK_FILES.get(book_id)

    if pdf and pdf.exists():
        print(f"Original PDF: {pdf}")
        print(f"Navigate to PDF page {page}.")

        if input("Open original PDF? (y/n): ").strip().lower() in ("y", "yes"):
            os.startfile(str(pdf))

    print("\nCompare EVERY word with the scanned page.")
    print("Approval confirms transcription accuracy only.")
    decision = input(
        "Decision: [a]pprove / [r]eject / [s]kip: "
    ).strip().lower()

    if decision not in ("a", "r"):
        print("Skipped. Status unchanged.")
        return

    approved = decision == "a"
    review_status = "APPROVED" if approved else "REJECTED"
    correction_status = "VERIFIED" if approved else "REJECTED"
    digest = text_hash(corrected)

    with conn:
        conn.execute("""
            INSERT INTO rag_reviews
                (chunk_id, decision, reviewed_hash)
            VALUES (?, ?, ?)
            ON CONFLICT(chunk_id) DO UPDATE SET
                decision = excluded.decision,
                reviewed_hash = excluded.reviewed_hash,
                reviewed_at = CURRENT_TIMESTAMP
        """, (chunk_id, review_status, digest))

        conn.execute("""
            UPDATE rag_corrections
            SET status = ?
            WHERE chunk_id = ?
        """, (correction_status, chunk_id))

        conn.execute("""
            UPDATE rag_chunks
            SET verified = ?
            WHERE id = ?
        """, (int(approved), chunk_id))

    print(f"Review saved: {review_status}")
    print("Clinical approval: NOT ASSESSED")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunk", type=int, required=True)
    args = parser.parse_args()

    with sqlite3.connect(DB_PATH) as conn:
        setup_database(conn)
        review_chunk(conn, args.chunk)


if __name__ == "__main__":
    main()
