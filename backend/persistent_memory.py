"""Local SQLite memory for VedAmrit. No external dependencies."""
import re
import sqlite3
from pathlib import Path

DEFAULT_DB = Path(__file__).resolve().parent.parent / 'storage' / 'vedamrit_memory.sqlite3'

class PersistentMemory:
    def __init__(self, db_path=None):
        self.db_path = Path(db_path) if db_path else DEFAULT_DB
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as db:
            db.execute('CREATE TABLE IF NOT EXISTS facts (key TEXT PRIMARY KEY, value TEXT NOT NULL)')

    def _connect(self):
        return sqlite3.connect(str(self.db_path), timeout=10)

    def set(self, key, value):
        with self._connect() as db:
            db.execute('INSERT INTO facts(key,value) VALUES (?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value', (key, value))

    def get(self, key):
        with self._connect() as db:
            row = db.execute('SELECT value FROM facts WHERE key=?', (key,)).fetchone()
        return row[0] if row else None

    def delete(self, key):
        with self._connect() as db:
            db.execute('DELETE FROM facts WHERE key=?', (key,))

    def clear(self):
        with self._connect() as db:
            db.execute('DELETE FROM facts')

    def all(self):
        with self._connect() as db:
            return dict(db.execute('SELECT key,value FROM facts ORDER BY key').fetchall())

    @staticmethod
    def extract_preference(text):
        text = text.strip()
        # Only explicit first-person affirmative declarations; not questions or negations.
        match = re.fullmatch(r'(?:मुझे|मेरे को)\s+(.{1,70}?)\s+पसंद\s+है[।.!\s]*', text)
        if not match:
            return None
        thing = match.group(1).strip()
        if not thing or any(w in thing for w in ('क्या', 'कौन', 'नहीं', 'मत', 'नही')):
            return None
        return thing
