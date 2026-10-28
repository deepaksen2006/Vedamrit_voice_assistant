
from pathlib import Path
import sqlite3
import argparse
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "storage" / "ayurveda_rag.db"
EDIT_DIR = ROOT / "storage" / "rag_corrections"

EDIT_DIR.mkdir(parents=True, exist_ok=True)


def setup(conn):
    conn.execute("""
        CREATE TABLE IF NOT EXISTS rag_corrections (
            chunk_id INTEGER PRIMARY KEY,
            corrected_text TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'PENDING',
            updated_at TEXT NOT NULL,
            FOREIGN KEY(chunk_id) REFERENCES rag_chunks(id)
        )
    """)


def export_chunk(conn, chunk_id):
    row = conn.execute(
        "SELECT text FROM rag_chunks WHERE id = ?",
        (chunk_id,)
    ).fetchone()

    if not row:
        print("Chunk not found.")
        return

    output = EDIT_DIR / f"chunk_{chunk_id}.txt"

    if output.exists():
        print(f"File already exists: {output}")
        print("Edit the existing file or move it before exporting.")
        return

    output.write_text(row[0], encoding="utf-8")

    print(f"Exported: {output}")
    print("Open this file in VS Code and correct the OCR.")


def import_chunk(conn, chunk_id):
    path = EDIT_DIR / f"chunk_{chunk_id}.txt"

    if not path.exists():
        print("Correction file not found.")
        return

    text = path.read_text(encoding="utf-8").strip()

    if not text:
        print("Correction file is empty.")
        return

    exists = conn.execute(
        "SELECT 1 FROM rag_chunks WHERE id = ?",
        (chunk_id,)
    ).fetchone()

    if not exists:
        print("Chunk not found.")
        return

    timestamp = datetime.now(timezone.utc).isoformat()

    with conn:
        conn.execute("""
            INSERT INTO rag_corrections
                (chunk_id, corrected_text, status, updated_at)
            VALUES (?, ?, 'PENDING', ?)
            ON CONFLICT(chunk_id) DO UPDATE SET
                corrected_text = excluded.corrected_text,
                status = 'PENDING',
                updated_at = excluded.updated_at
        """, (chunk_id, text, timestamp))

        # Any previous approval is invalid after an edit.
        conn.execute(
            "UPDATE rag_chunks SET verified = 0 WHERE id = ?",
            (chunk_id,)
        )

        conn.execute(
            "DELETE FROM rag_reviews WHERE chunk_id = ?",
            (chunk_id,)
        )

    print("Correction imported successfully.")
    print("Status: PENDING — requires manual verification.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunk", type=int, required=True)
    parser.add_argument(
        "--action",
        choices=["export", "import"],
        required=True
    )
    args = parser.parse_args()

    with sqlite3.connect(DB_PATH) as conn:
        setup(conn)

        if args.action == "export":
            export_chunk(conn, args.chunk)
        else:
            import_chunk(conn, args.chunk)


if __name__ == "__main__":
    main()
