
from pathlib import Path
from difflib import SequenceMatcher
import argparse
import json

ROOT = Path(__file__).resolve().parent.parent

MANUAL_DIR = ROOT / "storage" / "rag_corrections"
AI_DIR = ROOT / "storage" / "ai_ocr_suggestions"
REVIEW_DIR = ROOT / "storage" / "ai_ocr_reviews"

REVIEW_DIR.mkdir(parents=True, exist_ok=True)


def review(chunk_id):
    original_path = MANUAL_DIR / f"chunk_{chunk_id}.txt"
    suggestion_path = (
        AI_DIR / f"chunk_{chunk_id}_suggestion.txt"
    )

    if not original_path.exists():
        print("Original correction file not found.")
        return

    if not suggestion_path.exists():
        print("AI suggestion file not found.")
        return

    original = original_path.read_text(encoding="utf-8")
    suggested = suggestion_path.read_text(encoding="utf-8")

    # Preserve whitespace by tokenizing it separately.
    import re
    pattern = r"\s+|[^\s]+"
    original_tokens = re.findall(pattern, original)
    suggested_tokens = re.findall(pattern, suggested)

    matcher = SequenceMatcher(
        None,
        original_tokens,
        suggested_tokens,
        autojunk=False
    )

    operations = matcher.get_opcodes()

    result = []
    decisions = []

    print("\nAI OCR Correction Review")
    print("=" * 60)
    print(f"Chunk: {chunk_id}")
    print("Compare each suggestion with the original PDF.")
    print("=" * 60)

    for tag, i1, i2, j1, j2 in operations:
        old_text = "".join(original_tokens[i1:i2])
        new_text = "".join(suggested_tokens[j1:j2])

        if tag == "equal":
            result.append(old_text)
            continue

        print("\n" + "-" * 50)
        print("Original:")
        print(repr(old_text))
        print("\nAI suggestion:")
        print(repr(new_text))

        while True:
            choice = input(
                "\n[a]ccept / [r]eject / [q]uit: "
            ).strip().lower()

            if choice in ("a", "r", "q"):
                break

            print("Enter a, r, or q.")

        if choice == "q":
            print("Review cancelled. No files saved.")
            return

        accepted = choice == "a"

        result.append(
            new_text if accepted else old_text
        )

        decisions.append({
            "operation": tag,
            "original": old_text,
            "suggested": new_text,
            "decision": "ACCEPTED" if accepted else "REJECTED"
        })

    output_path = (
        REVIEW_DIR / f"chunk_{chunk_id}_reviewed.txt"
    )

    audit_path = (
        REVIEW_DIR / f"chunk_{chunk_id}_audit.json"
    )

    output_path.write_text(
        "".join(result),
        encoding="utf-8"
    )

    audit_path.write_text(
        json.dumps(
            decisions,
            ensure_ascii=False,
            indent=2
        ),
        encoding="utf-8"
    )

    print("\n" + "=" * 60)
    print("Review completed.")
    print(f"Changes reviewed: {len(decisions)}")
    print(f"Accepted: {sum(d['decision'] == 'ACCEPTED' for d in decisions)}")
    print(f"Rejected: {sum(d['decision'] == 'REJECTED' for d in decisions)}")
    print(f"Reviewed text: {output_path}")
    print(f"Audit log: {audit_path}")
    print("\nStatus: NOT VERIFIED against the complete PDF.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunk", type=int, required=True)
    args = parser.parse_args()

    review(args.chunk)


if __name__ == "__main__":
    main()
