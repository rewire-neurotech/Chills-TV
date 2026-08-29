"""Lightweight auth: a persistent visitor cookie for the profile hub (no
signup/password -- matches the low-friction flow of the rest of the app),
and a simple password-gated session for /admin.
"""
import os
import hmac

import db

VISITOR_COOKIE = "cv_token"
ADMIN_COOKIE = "cv_admin"

ADMIN_EMAIL = os.getenv("ADMIN_EMAIL", "")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "")


def get_current_user(request):
    token = request.cookies.get(VISITOR_COOKIE, "")
    return db.get_user_by_token(token)


def check_admin_login(email: str, password: str) -> bool:
    if not ADMIN_EMAIL or not ADMIN_PASSWORD:
        return False
    email_ok = hmac.compare_digest((email or "").strip().lower(), ADMIN_EMAIL.strip().lower())
    password_ok = hmac.compare_digest(password or "", ADMIN_PASSWORD)
    return email_ok and password_ok


def is_admin(request) -> bool:
    token = request.cookies.get(ADMIN_COOKIE, "")
    return db.admin_session_valid(token)
