"""Auth for ChillsTV / rewire.bio.

Three layers, oldest first:
1. The legacy visitor cookie (cv_token) -- anonymous users row, still fully
   supported so every existing user and /me /b /us link keeps working.
2. Account sessions (cv_session) -- email+password or Google accounts on top
   of the same users table, stored in auth_sessions. The cookie is set on the
   parent domain when running on rewire.bio, so app.rewire.bio (Edge) can
   later validate the same session.
3. The password-gated /admin session, unchanged.

Passwords are hashed with pbkdf2-sha256 (stdlib, no new dependency).
"""
import os
import hmac
import hashlib
import secrets

import db

VISITOR_COOKIE = "cv_token"
SESSION_COOKIE = "cv_session"
ADMIN_COOKIE = "cv_admin"

ADMIN_EMAIL = os.getenv("ADMIN_EMAIL", "")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "")

SESSION_TTL = 60 * 60 * 24 * 365

# the cookie is shared across subdomains only on the real domain, so that the
# same sign in works on app.rewire.bio. Everywhere else (localhost, onrender)
# a plain host cookie keeps local testing working.
COOKIE_DOMAIN = os.getenv("SESSION_COOKIE_DOMAIN", ".rewire.bio")

PBKDF2_ITERATIONS = 260000


# -- passwords ---------------------------------------------------------
def hash_password(password: str) -> str:
    salt = secrets.token_hex(16)
    dk = hashlib.pbkdf2_hmac("sha256", (password or "").encode(), salt.encode(), PBKDF2_ITERATIONS)
    return f"pbkdf2${PBKDF2_ITERATIONS}${salt}${dk.hex()}"

def verify_password(password: str, stored: str) -> bool:
    try:
        scheme, iters, salt, hexhash = (stored or "").split("$", 3)
        if scheme != "pbkdf2":
            return False
        dk = hashlib.pbkdf2_hmac("sha256", (password or "").encode(), salt.encode(), int(iters))
        return hmac.compare_digest(dk.hex(), hexhash)
    except (ValueError, TypeError):
        return False


# -- account sessions --------------------------------------------------
def secure_cookies(request) -> bool:
    """True when the request arrived over https. Render terminates TLS and
    forwards the scheme in x-forwarded-proto. Plain http local testing keeps
    working because the flag is simply left off there."""
    proto = (request.headers.get("x-forwarded-proto") or "").lower()
    if proto:
        return proto == "https"
    try:
        return (request.url.scheme or "").lower() == "https"
    except Exception:
        return True

def _use_domain(request) -> bool:
    host = (request.headers.get("host", "") or "").split(":")[0].lower()
    root = COOKIE_DOMAIN.lstrip(".")
    return host == root or host.endswith("." + root)

def start_session(response, request, user_id: int):
    """Create an auth session and set both cookies: the session cookie for
    the account, and the legacy visitor cookie pointed at the same users row
    so every v49 endpoint sees the same person."""
    token = db.create_auth_session(user_id, SESSION_TTL)
    kw = dict(max_age=SESSION_TTL, httponly=True, samesite="lax", secure=secure_cookies(request))
    if _use_domain(request):
        kw["domain"] = COOKIE_DOMAIN
    response.set_cookie(SESSION_COOKIE, token, **kw)
    user = db.get_user_by_id(user_id)
    if user:
        response.set_cookie(VISITOR_COOKIE, user["token"], max_age=SESSION_TTL,
                            httponly=True, samesite="lax", secure=secure_cookies(request))
    return token

def end_session(response, request):
    token = request.cookies.get(SESSION_COOKIE, "")
    db.revoke_auth_session(token)
    response.delete_cookie(SESSION_COOKIE)
    if _use_domain(request):
        response.delete_cookie(SESSION_COOKIE, domain=COOKIE_DOMAIN)
    response.delete_cookie(VISITOR_COOKIE)

def get_session_user(request):
    """The users row for a signed-in account session, or None."""
    sess = db.get_auth_session(request.cookies.get(SESSION_COOKIE, ""))
    if not sess:
        return None
    return db.get_user_by_id(sess["user_id"])


# -- current user (session first, legacy cookie second) ----------------
def get_current_user(request):
    user = get_session_user(request)
    if user:
        return user
    token = request.cookies.get(VISITOR_COOKIE, "")
    return db.get_user_by_token(token)

def is_logged_in(request) -> bool:
    return get_session_user(request) is not None


# -- admin (unchanged) -------------------------------------------------
def check_admin_login(email: str, password: str) -> bool:
    if not ADMIN_EMAIL or not ADMIN_PASSWORD:
        return False
    email_ok = hmac.compare_digest((email or "").strip().lower(), ADMIN_EMAIL.strip().lower())
    password_ok = hmac.compare_digest(password or "", ADMIN_PASSWORD)
    return email_ok and password_ok

def is_admin(request) -> bool:
    token = request.cookies.get(ADMIN_COOKIE, "")
    return db.admin_session_valid(token)
