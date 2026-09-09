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
    answers_json TEXT DEFAULT '{}',
    pending_send_token TEXT DEFAULT '',
    email TEXT DEFAULT '',
    password_hash TEXT DEFAULT '',
    google_sub TEXT DEFAULT '',
    consented_at REAL,
    beta_status TEXT DEFAULT 'none',
    pending_after_sid TEXT DEFAULT '',
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
    completed_at REAL,
    opened_at REAL
);

CREATE TABLE IF NOT EXISTS video_watches (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    stimulus_id TEXT NOT NULL,
    watched_at REAL NOT NULL,
    UNIQUE(user_id, stimulus_id)
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
    created_at REAL NOT NULL,
    user_id INTEGER
);

CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER,
    pid TEXT DEFAULT '',
    event TEXT NOT NULL,
    detail TEXT DEFAULT '{}',
    created_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS admin_sessions (
    token TEXT PRIMARY KEY,
    created_at REAL NOT NULL,
    expires_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS after_answers (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    stimulus_id TEXT NOT NULL,
    chills INTEGER NOT NULL,
    what_text TEXT DEFAULT '',
    why_text TEXT DEFAULT '',
    created_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS auth_sessions (
    token TEXT PRIMARY KEY,
    user_id INTEGER NOT NULL,
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
        conn.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_users_email ON users(email) WHERE email != ''"
        )


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

    existing = {row["name"] for row in conn.execute("PRAGMA table_info(users)")}
    if "answers_json" not in existing:
        conn.execute("ALTER TABLE users ADD COLUMN answers_json TEXT DEFAULT '{}'")
    for col, decl in [
        ("email", "TEXT DEFAULT ''"),
        ("password_hash", "TEXT DEFAULT ''"),
        ("google_sub", "TEXT DEFAULT ''"),
        ("consented_at", "REAL"),
        ("beta_status", "TEXT DEFAULT 'none'"),
        ("pending_after_sid", "TEXT DEFAULT ''"),
    ]:
        if col not in existing:
            conn.execute(f"ALTER TABLE users ADD COLUMN {col} {decl}")

    existing = {row["name"] for row in conn.execute("PRAGMA table_info(duo_pairs)")}
    if "opened_at" not in existing:
        conn.execute("ALTER TABLE duo_pairs ADD COLUMN opened_at REAL")

    existing = {row["name"] for row in conn.execute("PRAGMA table_info(video_comments)")}
    if "user_id" not in existing:
        conn.execute("ALTER TABLE video_comments ADD COLUMN user_id INTEGER")


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
                       top5_json: str = None, vector_json: str = None,
                       answers_json: str = None):
    with get_conn() as conn:
        conn.execute(
            """UPDATE users SET stimulus_id=?, stimulus_name=?, stimulus_url=?,
               score=?, percentile=?, paid=?,
               top5_json=COALESCE(?, top5_json), vector_json=COALESCE(?, vector_json),
               answers_json=COALESCE(?, answers_json)
               WHERE token=?""",
            (stimulus_id, stimulus_name, stimulus_url, score, percentile, int(paid),
             top5_json, vector_json, answers_json, token),
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


def mark_duo_opened(token: str):
    """First time the partner opens the duo link. Only stamps once."""
    with get_conn() as conn:
        conn.execute(
            "UPDATE duo_pairs SET opened_at=? WHERE token=? AND opened_at IS NULL",
            (time.time(), token),
        )


def duos_for_user(user_id: int):
    """All duo pairs this user initiated, newest first. Feeds the results
    and waiting lists on the duo intro page."""
    with get_conn() as conn:
        return conn.execute(
            "SELECT * FROM duo_pairs WHERE user_id=? ORDER BY created_at DESC",
            (user_id,),
        ).fetchall()


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
def add_video_comment(stimulus_id: str, text: str, experienced: bool, author: str = "You",
                       user_id: int = None):
    with get_conn() as conn:
        conn.execute(
            """INSERT INTO video_comments (stimulus_id, author, text, experienced, created_at, user_id)
               VALUES (?,?,?,?,?,?)""",
            (stimulus_id, author, text.strip(), int(experienced), time.time(), user_id),
        )


def comments_for(stimulus_id: str, limit: int = 20):
    with get_conn() as conn:
        return conn.execute(
            """SELECT * FROM video_comments WHERE stimulus_id=?
               ORDER BY created_at DESC LIMIT ?""",
            (stimulus_id, limit),
        ).fetchall()


def comments_by_user(user_id: int, limit: int = 50):
    """A user's comments, newest first. Feeds the expandable rows in admin."""
    with get_conn() as conn:
        return conn.execute(
            """SELECT * FROM video_comments WHERE user_id=?
               ORDER BY created_at DESC LIMIT ?""",
            (user_id, limit),
        ).fetchall()


# ── video watches ─────────────────────────────────────────────────────
def record_watch(user_id: int, stimulus_id: str):
    """One row per user per video, first watch wins."""
    if not user_id or not stimulus_id:
        return
    with get_conn() as conn:
        conn.execute(
            "INSERT OR IGNORE INTO video_watches (user_id, stimulus_id, watched_at) VALUES (?,?,?)",
            (user_id, stimulus_id, time.time()),
        )


def has_watched(user_id: int, stimulus_id: str) -> bool:
    if not user_id or not stimulus_id:
        return False
    with get_conn() as conn:
        row = conn.execute(
            "SELECT 1 FROM video_watches WHERE user_id=? AND stimulus_id=?",
            (user_id, stimulus_id),
        ).fetchone()
    return bool(row)


def watched_counts_map() -> dict:
    """user_id -> number of distinct videos watched. For the admin table."""
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT user_id, COUNT(*) AS n FROM video_watches GROUP BY user_id"
        ).fetchall()
    return {r["user_id"]: r["n"] for r in rows}


def sent_counts_map() -> dict:
    """user_id -> number of sends created. For the admin table."""
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT sender_user_id AS user_id, COUNT(*) AS n FROM sends GROUP BY sender_user_id"
        ).fetchall()
    return {r["user_id"]: r["n"] for r in rows}


# ── admin series ──────────────────────────────────────────────────────
def daily_counts(table: str) -> list:
    """Rows of (day 'YYYY-MM-DD', count) ascending, for the growth charts.
    Table name is whitelisted, never interpolated from user input."""
    assert table in ("users", "sends", "duo_pairs")
    with get_conn() as conn:
        return conn.execute(
            f"""SELECT date(created_at, 'unixepoch') AS day, COUNT(*) AS n
                FROM {table} GROUP BY day ORDER BY day ASC"""
        ).fetchall()


def count_since(table: str, ts: float) -> int:
    """Rows created at or after ts. For the '+N today' tiles."""
    assert table in ("users", "sends", "duo_pairs")
    with get_conn() as conn:
        row = conn.execute(
            f"SELECT COUNT(*) AS n FROM {table} WHERE created_at >= ?", (ts,)
        ).fetchone()
    return int(row["n"])


def count_sends_experienced() -> int:
    """Sends where the recipient reported chills. For the reported-chills tile."""
    with get_conn() as conn:
        row = conn.execute(
            "SELECT COUNT(*) AS n FROM sends WHERE experienced = 1"
        ).fetchone()
    return int(row["n"])


# ── events (full action log) ──────────────────────────────────────────
def log_event(event: str, user_id: int = None, pid: str = "", detail: str = "{}"):
    """One row per user action. Never raises, a logging failure must not
    break the request it rides on."""
    try:
        with get_conn() as conn:
            conn.execute(
                "INSERT INTO events (user_id, pid, event, detail, created_at) VALUES (?,?,?,?,?)",
                (user_id, pid or "", event, detail or "{}", time.time()),
            )
    except Exception:
        pass


def all_events(limit: int = 200000):
    with get_conn() as conn:
        return conn.execute(
            "SELECT * FROM events ORDER BY created_at ASC LIMIT ?", (limit,)
        ).fetchall()


# ── accounts (v50 unification) ───────────────────────────────────────
def get_user_by_email(email: str):
    email = (email or "").strip().lower()
    if not email:
        return None
    with get_conn() as conn:
        return conn.execute("SELECT * FROM users WHERE email=?", (email,)).fetchone()


def get_user_by_google_sub(sub: str):
    if not sub:
        return None
    with get_conn() as conn:
        return conn.execute("SELECT * FROM users WHERE google_sub=?", (sub,)).fetchone()


def attach_account(user_token: str, email: str, password_hash: str = "", google_sub: str = ""):
    """Bind an email account to an existing cookie user row, so a visitor who
    already took the test keeps their profile when they sign up."""
    with get_conn() as conn:
        conn.execute(
            """UPDATE users SET email=?, password_hash=COALESCE(NULLIF(?, ''), password_hash),
               google_sub=COALESCE(NULLIF(?, ''), google_sub) WHERE token=?""",
            ((email or "").strip().lower(), password_hash, google_sub, user_token),
        )
        return conn.execute("SELECT * FROM users WHERE token=?", (user_token,)).fetchone()


def create_account_user(email: str, password_hash: str = "", google_sub: str = "") -> sqlite3.Row:
    """A fresh users row that starts life with an account attached. The row
    still gets a token so every legacy flow (hub, sends, duo) works on it."""
    token = new_token(16)
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO users (token, email, password_hash, google_sub, created_at) VALUES (?,?,?,?,?)",
            (token, (email or "").strip().lower(), password_hash, google_sub, time.time()),
        )
        return conn.execute("SELECT * FROM users WHERE token=?", (token,)).fetchone()


def set_password(user_id: int, password_hash: str):
    with get_conn() as conn:
        conn.execute("UPDATE users SET password_hash=? WHERE id=?", (password_hash, user_id))


def set_consented(user_id: int):
    with get_conn() as conn:
        conn.execute(
            "UPDATE users SET consented_at=? WHERE id=? AND consented_at IS NULL",
            (time.time(), user_id),
        )


def set_beta_status(user_id: int, status: str):
    assert status in ("none", "listed", "granted")
    with get_conn() as conn:
        conn.execute("UPDATE users SET beta_status=? WHERE id=?", (status, user_id))


def set_pending_after(user_id: int, stimulus_id: str):
    with get_conn() as conn:
        conn.execute("UPDATE users SET pending_after_sid=? WHERE id=?", (stimulus_id or "", user_id))


def clear_pending_after(user_id: int):
    with get_conn() as conn:
        conn.execute("UPDATE users SET pending_after_sid='' WHERE id=?", (user_id,))


# ── after the video answers ──────────────────────────────────
def record_after_answers(user_id: int, stimulus_id: str, chills: bool,
                          what_text: str = "", why_text: str = ""):
    if not user_id or not stimulus_id:
        return
    with get_conn() as conn:
        conn.execute(
            """INSERT INTO after_answers (user_id, stimulus_id, chills, what_text, why_text, created_at)
               VALUES (?,?,?,?,?,?)""",
            (user_id, stimulus_id, int(bool(chills)), (what_text or "").strip(),
             (why_text or "").strip(), time.time()),
        )


def after_answers_for_user(user_id: int, limit: int = 100):
    with get_conn() as conn:
        return conn.execute(
            "SELECT * FROM after_answers WHERE user_id=? ORDER BY created_at DESC LIMIT ?",
            (user_id, limit),
        ).fetchall()


# ── auth sessions (account sign in) ───────────────────────────
def create_auth_session(user_id: int, ttl_seconds: int = 60 * 60 * 24 * 365) -> str:
    token = new_token(24)
    now = time.time()
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO auth_sessions (token, user_id, created_at, expires_at) VALUES (?,?,?,?)",
            (token, user_id, now, now + ttl_seconds),
        )
    return token


def get_auth_session(token: str):
    """The session row if the token is valid and unexpired, else None."""
    if not token:
        return None
    with get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM auth_sessions WHERE token=?", (token,)
        ).fetchone()
    if row and row["expires_at"] > time.time():
        return row
    return None


def revoke_auth_session(token: str):
    if not token:
        return
    with get_conn() as conn:
        conn.execute("DELETE FROM auth_sessions WHERE token=?", (token,))


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
