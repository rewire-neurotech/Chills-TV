"""SQLite persistence for the ChillsTV v38 features (profile hub, Send Chills,
Duo compatibility, contributions, admin auth). Keeps everything the ONNX/Stripe
core in app.py already does untouched -- this is purely additive.
"""
import os
import sqlite3
import time
import secrets
from contextlib import contextmanager

DB_DIR = "/data" if os.path.isdir("/data") else os.path.dirname(__file__)
DB_PATH = os.path.join(DB_DIR, "chillstv.db")

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    token TEXT UNIQUE NOT NULL,
    pid TEXT DEFAULT '',
    display_name TEXT DEFAULT '',
    stimulus_id TEXT DEFAULT '',
    stimulus_name TEXT DEFAULT '',
    stimulus_url TEXT DEFAULT '',
    score REAL DEFAULT 0,
    percentile REAL DEFAULT 0,
    paid INTEGER DEFAULT 0,
    session_id TEXT DEFAULT '',
    top5_json TEXT DEFAULT '[]',
    vector_json TEXT DEFAULT '[]',
    pending_send_token TEXT DEFAULT '',
    created_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS sends (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    token TEXT UNIQUE NOT NULL,
    sender_user_id INTEGER NOT NULL,
    recipient_name TEXT DEFAULT '',
    stimulus_id TEXT DEFAULT '',
    stimulus_name TEXT DEFAULT '',
    stimulus_url TEXT DEFAULT '',
    mode TEXT DEFAULT 'picked',
    status TEXT DEFAULT 'sent',
    experienced INTEGER,
    intensity INTEGER,
    chills_length INTEGER,
    chills_waves INTEGER,
    description TEXT DEFAULT '',
    closeness INTEGER,
    relationship TEXT,
    sender_seen INTEGER DEFAULT 0,
    created_at REAL NOT NULL,
    watched_at REAL,
    revealed_at REAL
);

CREATE TABLE IF NOT EXISTS duo_pairs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    token TEXT UNIQUE NOT NULL,
    user_id INTEGER NOT NULL,
    partner_user_id INTEGER,
    partner_name TEXT DEFAULT '',
    status TEXT DEFAULT 'pending',
    match_pct REAL,
    video_stimulus_id TEXT,
    video_stimulus_name TEXT,
    created_at REAL NOT NULL,
    completed_at REAL
);

CREATE TABLE IF NOT EXISTS contributions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    url TEXT DEFAULT '',
    description TEXT DEFAULT '',
    submitted_by INTEGER,
    status TEXT DEFAULT 'pending',
    created_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS video_comments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    stimulus_id TEXT NOT NULL,
    author TEXT DEFAULT 'Anonymous',
    text TEXT NOT NULL,
    experienced INTEGER DEFAULT 0,
    created_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS admin_sessions (
    token TEXT PRIMARY KEY,
    created_at REAL NOT NULL,
    expires_at REAL NOT NULL
);
"""


@contextmanager
def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    os.makedirs(DB_DIR, exist_ok=True)
    with get_conn() as conn:
        conn.executescript(SCHEMA)
        _migrate(conn)


def _migrate(conn):
    """CREATE TABLE IF NOT EXISTS doesn't add new columns to an existing table
    (e.g. on Render's already-populated database), so new columns are added
    here, guarded against already existing."""
    existing = {row["name"] for row in conn.execute("PRAGMA table_info(sends)")}
    for col, decl in [
        ("chills_length", "INTEGER"),
        ("chills_waves", "INTEGER"),
        ("description", "TEXT DEFAULT ''"),
    ]:
        if col not in existing:
            conn.execute(f"ALTER TABLE sends ADD COLUMN {col} {decl}")


def new_token(nbytes: int = 8) -> str:
    return secrets.token_urlsafe(nbytes)


# ── users ──────────────────────────────────────────────────────────────
def create_user(pid: str = "", session_id: str = "") -> sqlite3.Row:
    token = new_token(16)
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO users (token, pid, session_id, created_at) VALUES (?,?,?,?)",
            (token, pid, session_id, time.time()),
        )
        return conn.execute("SELECT * FROM users WHERE token = ?", (token,)).fetchone()


def get_user_by_token(token: str):
    if not token:
        return None
    with get_conn() as conn:
        return conn.execute("SELECT * FROM users WHERE token = ?", (token,)).fetchone()


def get_user_by_id(user_id: int):
    if not user_id:
        return None
    with get_conn() as conn:
        return conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()


def update_user_match(token: str, stimulus_id: str, stimulus_name: str, stimulus_url: str,
                       score: float, percentile: float, paid: bool = True,
                       top5_json: str = None, vector_json: str = None):
    with get_conn() as conn:
        conn.execute(
            """UPDATE users SET stimulus_id=?, stimulus_name=?, stimulus_url=?,
               score=?, percentile=?, paid=?,
               top5_json=COALESCE(?, top5_json), vector_json=COALESCE(?, vector_json)
               WHERE token=?""",
            (stimulus_id, stimulus_name, stimulus_url, score, percentile, int(paid),
             top5_json, vector_json, token),
        )


def set_percentile(token: str, percentile: float):
    with get_conn() as conn:
        conn.execute("UPDATE users SET percentile=? WHERE token=?", (percentile, token))


def set_pending_send_token(user_token: str, send_token: str):
    with get_conn() as conn:
        conn.execute("UPDATE users SET pending_send_token=? WHERE token=?", (send_token, user_token))


def count_users() -> int:
    with get_conn() as conn:
        return conn.execute("SELECT COUNT(*) c FROM users").fetchone()["c"]


def count_paid_users() -> int:
    with get_conn() as conn:
        return conn.execute("SELECT COUNT(*) c FROM users WHERE paid=1").fetchone()["c"]


def all_scores() -> list:
    with get_conn() as conn:
        rows = conn.execute("SELECT score FROM users WHERE paid=1").fetchall()
    return [r["score"] for r in rows]


def percentile_for_score(score: float) -> float:
    """Share of paid users this score beats or ties, as a rank percentile."""
    scores = all_scores()
    if len(scores) < 2:
        return 50.0
    below_or_eq = sum(1 for s in scores if s <= score)
    return round(100.0 * below_or_eq / len(scores), 1)


def all_users(order_by: str = "created_at", desc: bool = True, limit: int = 1000, offset: int = 0):
    order_by = order_by if order_by in ("created_at", "score") else "created_at"
    direction = "DESC" if desc else "ASC"
    with get_conn() as conn:
        return conn.execute(
            f"SELECT * FROM users ORDER BY {order_by} {direction} LIMIT ? OFFSET ?",
            (limit, offset),
        ).fetchall()


# ── sends (Send Chills) ───────────────────────────────────────────────
def create_send(sender_user_id: int, stimulus_id: str, stimulus_name: str,
                 stimulus_url: str, mode: str = "picked") -> sqlite3.Row:
    token = new_token(6)
    with get_conn() as conn:
        conn.execute(
            """INSERT INTO sends (token, sender_user_id, stimulus_id, stimulus_name,
               stimulus_url, mode, created_at) VALUES (?,?,?,?,?,?,?)""",
            (token, sender_user_id, stimulus_id, stimulus_name, stimulus_url, mode, time.time()),
        )
        return conn.execute("SELECT * FROM sends WHERE token=?", (token,)).fetchone()


def set_send_stimulus(token: str, stimulus_id: str, stimulus_name: str, stimulus_url: str):
    with get_conn() as conn:
        conn.execute(
            "UPDATE sends SET stimulus_id=?, stimulus_name=?, stimulus_url=? WHERE token=?",
            (stimulus_id, stimulus_name, stimulus_url, token),
        )


def get_send_by_token(token: str):
    with get_conn() as conn:
        return conn.execute("SELECT * FROM sends WHERE token=?", (token,)).fetchone()


def sends_for_sender(sender_user_id: int):
    with get_conn() as conn:
        return conn.execute(
            "SELECT * FROM sends WHERE sender_user_id=? ORDER BY created_at DESC",
            (sender_user_id,),
        ).fetchall()


def record_send_response(token: str, experienced: bool, intensity: int = 0,
                          recipient_name: str = "", chills_length: int = None,
                          chills_waves: int = None, description: str = ""):
    with get_conn() as conn:
        conn.execute(
            """UPDATE sends SET status='watched', experienced=?, intensity=?,
               chills_length=?, chills_waves=?, description=?,
               recipient_name=COALESCE(NULLIF(?, ''), recipient_name), watched_at=?
               WHERE token=?""",
            (int(experienced), intensity, chills_length, chills_waves, description,
             recipient_name, time.time(), token),
        )


def record_reveal_gate(token: str, closeness: int, relationship: str):
    with get_conn() as conn:
        conn.execute(
            """UPDATE sends SET closeness=?, relationship=?, status='revealed',
               sender_seen=1, revealed_at=? WHERE token=?""",
            (closeness, relationship, time.time(), token),
        )


# ── duo (chills compatibility) ────────────────────────────────────────
def create_duo(user_id: int) -> sqlite3.Row:
    token = new_token(6)
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO duo_pairs (token, user_id, created_at) VALUES (?,?,?)",
            (token, user_id, time.time()),
        )
        return conn.execute("SELECT * FROM duo_pairs WHERE token=?", (token,)).fetchone()


def get_duo_by_token(token: str):
    with get_conn() as conn:
        return conn.execute("SELECT * FROM duo_pairs WHERE token=?", (token,)).fetchone()


def complete_duo(token: str, partner_user_id: int, partner_name: str, match_pct: float,
                  video_stimulus_id: str, video_stimulus_name: str):
    with get_conn() as conn:
        conn.execute(
            """UPDATE duo_pairs SET partner_user_id=?, partner_name=?, status='completed',
               match_pct=?, video_stimulus_id=?, video_stimulus_name=?, completed_at=?
               WHERE token=?""",
            (partner_user_id, partner_name, match_pct, video_stimulus_id,
             video_stimulus_name, time.time(), token),
        )


# ── contributions ──────────────────────────────────────────────────────
def create_contribution(url: str, description: str, submitted_by: int = None):
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO contributions (url, description, submitted_by, created_at) VALUES (?,?,?,?)",
            (url, description, submitted_by, time.time()),
        )


def all_contributions(limit: int = 200):
    with get_conn() as conn:
        return conn.execute(
            "SELECT * FROM contributions ORDER BY created_at DESC LIMIT ?", (limit,)
        ).fetchall()


# ── video comments ────────────────────────────────────────────────────
def add_video_comment(stimulus_id: str, text: str, experienced: bool, author: str = "You"):
    with get_conn() as conn:
        conn.execute(
            """INSERT INTO video_comments (stimulus_id, author, text, experienced, created_at)
               VALUES (?,?,?,?,?)""",
            (stimulus_id, author, text.strip(), int(experienced), time.time()),
        )


def comments_for(stimulus_id: str, limit: int = 20):
    with get_conn() as conn:
        return conn.execute(
            """SELECT * FROM video_comments WHERE stimulus_id=?
               ORDER BY created_at DESC LIMIT ?""",
            (stimulus_id, limit),
        ).fetchall()


# ── admin sessions ─────────────────────────────────────────────────────
def create_admin_session(ttl_seconds: int = 60 * 60 * 12) -> str:
    token = new_token(24)
    now = time.time()
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO admin_sessions (token, created_at, expires_at) VALUES (?,?,?)",
            (token, now, now + ttl_seconds),
        )
    return token


def admin_session_valid(token: str) -> bool:
    if not token:
        return False
    with get_conn() as conn:
        row = conn.execute(
            "SELECT expires_at FROM admin_sessions WHERE token=?", (token,)
        ).fetchone()
    return bool(row) and row["expires_at"] > time.time()


def revoke_admin_session(token: str):
    if not token:
        return
    with get_conn() as conn:
        conn.execute("DELETE FROM admin_sessions WHERE token=?", (token,))
