
import sqlite3
from pathlib import Path

# Database location
DB_PATH = Path(__file__).resolve().parent / "dermai.db"


def init_db():
    """Create or upgrade the prediction history table."""

    with sqlite3.connect(DB_PATH) as conn:

        conn.execute("""
            CREATE TABLE IF NOT EXISTS prediction_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                image_name TEXT NOT NULL,
                predicted_class TEXT NOT NULL,
                confidence REAL NOT NULL,
                melanoma_score REAL NOT NULL DEFAULT 0,
                nevus_score REAL NOT NULL DEFAULT 0,
                seborrheic_keratosis_score REAL NOT NULL DEFAULT 0,
                non_lesional_skin_score REAL NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL
                    DEFAULT (datetime('now', 'localtime'))
            )
        """)

        # Upgrade an existing database without deleting old records.
        columns = {
            row[1] for row in
            conn.execute("PRAGMA table_info(prediction_history)")
        }

        if "non_lesional_skin_score" not in columns:
            conn.execute("""
                ALTER TABLE prediction_history
                ADD COLUMN non_lesional_skin_score REAL NOT NULL DEFAULT 0
            """)


def save_prediction(
    image_name,
    predicted_class,
    confidence,
    melanoma_score,
    nevus_score,
    seborrheic_keratosis_score,
    non_lesional_skin_score=0.0
):
    """Save all four class scores as percentages."""

    init_db()

    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            INSERT INTO prediction_history (
                image_name,
                predicted_class,
                confidence,
                melanoma_score,
                nevus_score,
                seborrheic_keratosis_score,
                non_lesional_skin_score
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            image_name,
            predicted_class,
            confidence,
            melanoma_score,
            nevus_score,
            seborrheic_keratosis_score,
            non_lesional_skin_score
        ))


def get_prediction_history(limit=50):
    """Return recent prediction records."""

    init_db()

    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row

        rows = conn.execute("""
            SELECT *
            FROM prediction_history
            ORDER BY id DESC
            LIMIT ?
        """, (limit,)).fetchall()

        return [dict(row) for row in rows]


def get_all_prediction_history():
    """Return all prediction records."""

    init_db()

    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row

        rows = conn.execute("""
            SELECT *
            FROM prediction_history
            ORDER BY id DESC
        """).fetchall()

        return [dict(row) for row in rows]


def delete_prediction_history():
    """Delete all prediction records."""

    init_db()

    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.execute("DELETE FROM prediction_history")
        return max(cursor.rowcount, 0)
