"""SQLite storage. Everything stays in one local file; nothing leaves the machine."""

from __future__ import annotations

import hashlib
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import pandas as pd

from .normalize import normalize_description

DEFAULT_DB = Path(__file__).resolve().parent.parent / "data" / "zfinance.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS transactions (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    txn_hash      TEXT UNIQUE NOT NULL,
    date          TEXT NOT NULL,
    description   TEXT NOT NULL,
    clean_desc    TEXT NOT NULL,
    amount        REAL NOT NULL,
    account       TEXT DEFAULT '',
    category      TEXT,
    confidence    REAL,
    label_source  TEXT,          -- 'user' | 'model' | NULL (uncategorized)
    imported_at   TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_txn_date ON transactions(date);
CREATE INDEX IF NOT EXISTS idx_txn_label ON transactions(label_source);

CREATE TABLE IF NOT EXISTS budgets (
    category       TEXT PRIMARY KEY,
    monthly_limit  REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS model_runs (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    trained_at   TEXT NOT NULL,
    n_labels     INTEGER NOT NULL,
    cv_accuracy  REAL,
    cv_macro_f1  REAL
);
"""


class Store:
    def __init__(self, path: str | Path = DEFAULT_DB):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as c:
            c.executescript(SCHEMA)

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(self.path)
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    # ---------- transactions ----------

    @staticmethod
    def _hashes(df: pd.DataFrame) -> list[str]:
        """Stable per-row hash. The occurrence counter keeps genuine same-day duplicates
        (two $2.10 coffees) while still deduping re-imports of the same export."""
        keys = (
            df["date"].dt.strftime("%Y-%m-%d")
            + "|" + df["description"].str.upper()
            + "|" + df["amount"].map(lambda a: f"{a:.2f}")
            + "|" + df["account"].astype(str)
        )
        occ = keys.groupby(keys).cumcount().astype(str)
        return [hashlib.sha1(f"{k}|{o}".encode()).hexdigest() for k, o in zip(keys, occ)]

    def add_transactions(self, df: pd.DataFrame) -> tuple[int, int]:
        """Insert parsed transactions. Returns (inserted, skipped_as_duplicate)."""
        df = df.copy()
        df["txn_hash"] = self._hashes(df)
        now = datetime.now().isoformat(timespec="seconds")
        # Optional pre-labelled category column (used by the synthetic demo data).
        if "category" not in df.columns:
            df["category"] = None
        rows = []
        for r in df.itertuples():
            cat = r.category if isinstance(r.category, str) and r.category else None
            rows.append(
                (
                    r.txn_hash,
                    r.date.strftime("%Y-%m-%d"),
                    r.description,
                    normalize_description(r.description),
                    float(r.amount),
                    r.account or "",
                    cat,
                    1.0 if cat else None,
                    "user" if cat else None,
                    now,
                )
            )
        with self._conn() as c:
            before = c.execute("SELECT COUNT(*) FROM transactions").fetchone()[0]
            c.executemany(
                """INSERT OR IGNORE INTO transactions
                   (txn_hash, date, description, clean_desc, amount, account,
                    category, confidence, label_source, imported_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?)""",
                rows,
            )
            after = c.execute("SELECT COUNT(*) FROM transactions").fetchone()[0]
        inserted = after - before
        return inserted, len(rows) - inserted

    def transactions(self) -> pd.DataFrame:
        with self._conn() as c:
            df = pd.read_sql_query("SELECT * FROM transactions ORDER BY date DESC, id DESC", c)
        df["date"] = pd.to_datetime(df["date"])
        return df

    def set_labels(self, ids_to_category: dict[int, str]) -> None:
        with self._conn() as c:
            c.executemany(
                "UPDATE transactions SET category=?, label_source='user', confidence=1.0 WHERE id=?",
                [(cat, int(i)) for i, cat in ids_to_category.items()],
            )

    def set_predictions(self, ids: list[int], cats: list[str], confs: list[float]) -> None:
        """Write model predictions, never overwriting a user label."""
        with self._conn() as c:
            c.executemany(
                """UPDATE transactions SET category=?, confidence=?, label_source='model'
                   WHERE id=? AND (label_source IS NULL OR label_source='model')""",
                [(cat, float(p), int(i)) for i, cat, p in zip(ids, cats, confs)],
            )

    def clear(self) -> None:
        with self._conn() as c:
            c.execute("DELETE FROM transactions")
            c.execute("DELETE FROM model_runs")

    # ---------- budgets ----------

    def budgets(self) -> dict[str, float]:
        with self._conn() as c:
            return dict(c.execute("SELECT category, monthly_limit FROM budgets").fetchall())

    def set_budget(self, category: str, limit: float | None) -> None:
        with self._conn() as c:
            if limit is None or limit <= 0:
                c.execute("DELETE FROM budgets WHERE category=?", (category,))
            else:
                c.execute(
                    "INSERT INTO budgets(category, monthly_limit) VALUES(?,?) "
                    "ON CONFLICT(category) DO UPDATE SET monthly_limit=excluded.monthly_limit",
                    (category, float(limit)),
                )

    # ---------- model runs ----------

    def log_model_run(self, n_labels: int, acc: float | None, f1: float | None) -> None:
        with self._conn() as c:
            c.execute(
                "INSERT INTO model_runs(trained_at, n_labels, cv_accuracy, cv_macro_f1) VALUES(?,?,?,?)",
                (datetime.now().isoformat(timespec="seconds"), n_labels, acc, f1),
            )

    def model_runs(self) -> pd.DataFrame:
        with self._conn() as c:
            return pd.read_sql_query("SELECT * FROM model_runs ORDER BY id", c)
