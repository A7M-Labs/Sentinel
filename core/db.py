import sqlite3
from core.config import CFG

_db = sqlite3.connect(CFG.db_path, check_same_thread=False)
_db.execute("PRAGMA journal_mode=WAL")
_db.execute(
    """
    CREATE TABLE IF NOT EXISTS events(
      id INTEGER PRIMARY KEY,
      ts TEXT,
      label TEXT,
      score REAL,
      confidence REAL,
      start REAL,
      end REAL
    )
    """
)
_db.commit()


def record_event(label: str, score: float, confidence: float, start: float, end: float):
    _db.execute(
        "INSERT INTO events VALUES(NULL, datetime('now'), ?, ?, ?, ?, ?)" ,
        (label, score, confidence, start, end)
    )
    _db.commit()