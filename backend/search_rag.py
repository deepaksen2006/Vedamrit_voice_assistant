
from pathlib import Path
import sqlite3
import hashlib
import sys
import re

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "storage" / "ayurveda_rag.db"


def text_hash(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def search_books(query, limit=5, verified_only=True):
    if not DB_PATH.exists():
        raise FileNotFoundError("RAG database is missing.")

    if limit < 1:
        return []

    terms = re.findall(r"[\w\u0900-\u097F]+", query)
    if not terms:
        return []

    # Escape quoted FTS5 terms.
    expression = " OR ".join(
        '"' + term.replace('"', '""') + '"'
        for term in terms
    )

    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row

        # Search the original FTS index first.
        # Retrieve more candidates than needed because some may
        # not have approved, hash-matched corrections.
        candidates = conn.execute("""
            SELECT
                c.id,
                c.book_title,
                c.pdf_page,
                c.text AS original_text,
                c.verified,
                x.corrected_text,
                x.status AS correction_status,
                r.decision,
                r.reviewed_hash,
                bm25(rag_fts) AS score
            FROM rag_fts
            JOIN rag_chunks c ON c.id = rag_fts.rowid
            LEFT JOIN rag_corrections x ON x.chunk_id = c.id
            LEFT JOIN rag_reviews r ON r.chunk_id = c.id
            WHERE rag_fts MATCH ?
            ORDER BY score
        """, (expression,)).fetchall()

        # Also find approved corrections containing query terms.
        # This allows words absent from the original OCR index
        # to be discovered without modifying the FTS table.
        corrected_candidates = conn.execute("""
            SELECT
                c.id,
                c.book_title,
                c.pdf_page,
                c.text AS original_text,
                c.verified,
                x.corrected_text,
                x.status AS correction_status,
                r.decision,
                r.reviewed_hash,
                NULL AS score
            FROM rag_corrections x
            JOIN rag_chunks c ON c.id = x.chunk_id
            JOIN rag_reviews r ON r.chunk_id = c.id
            WHERE c.verified = 1
              AND x.status = 'VERIFIED'
              AND r.decision = 'APPROVED'
        """).fetchall()

    results = {}
    query_terms = [term.casefold() for term in terms]

    for row in list(candidates) + list(corrected_candidates):
        correction = row["corrected_text"]

        approved = (
            row["verified"] == 1
            and row["correction_status"] == "VERIFIED"
            and row["decision"] == "APPROVED"
            and correction is not None
            and row["reviewed_hash"] is not None
            and text_hash(correction) == row["reviewed_hash"]
        )

        if verified_only and not approved:
            continue

        effective_text = (
            correction if approved else row["original_text"]
        )

        # Match against the text that will actually be returned.
        if not any(
            term in effective_text.casefold()
            for term in query_terms
        ):
            continue

        # Simple term-coverage ranking for corrected text.
        matches = sum(
            term in effective_text.casefold()
            for term in query_terms
        )

        results[row["id"]] = {
            "chunk_id": row["id"],
            "book": row["book_title"],
            "page": row["pdf_page"],
            "text": effective_text,
            "verified": approved,
            "score": row["score"],
            "matches": matches,
            "citation": (
                f"{row['book_title']}, PDF page {row['pdf_page']}"
            )
        }

    # Prefer passages matching more query terms.
    # Use BM25 to break ties when available.
    ranked = sorted(
        results.values(),
        key=lambda item: (
            -item["matches"],
            item["score"] if item["score"] is not None else float("inf"),
            item["chunk_id"]
        )
    )

    return ranked[:limit]


if __name__ == "__main__":
    query = " ".join(sys.argv[1:]).strip()

    if not query:
        query = input("Enter Ayurvedic question: ").strip()

    results = search_books(query)

    if not results:
        print("No matching verified passages found.")

    for i, result in enumerate(results, 1):
        print(f"\nRESULT {i}")
        print("Chunk:", result["chunk_id"])
        print("Source:", result["citation"])
        print("Verified:", result["verified"])
        print("Text:", result["text"][:600])
