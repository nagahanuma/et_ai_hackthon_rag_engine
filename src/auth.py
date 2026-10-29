"""
Production Cryptographic Token Management (HMAC-SHA256 JWT).
Zero external dependencies - uses standard library hashlib, hmac, and base64.
"""

import hmac
import hashlib
import base64
import json
import time
from typing import Dict, Any, Optional
from src.config import settings
from src.logger import get_logger

logger = get_logger(__name__)

SECRET_KEY = getattr(settings, "SECRET_KEY", "enterprise_rag_secret_key_2026_signature")
TOKEN_EXPIRY_SECONDS = 7 * 24 * 3600  # 7 days


def _base64url_encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode("utf-8").rstrip("=")


def _base64url_decode(data: str) -> bytes:
    padding = "=" * (4 - len(data) % 4)
    return base64.urlsafe_b64decode((data + padding).encode("utf-8"))


def create_access_token(user_id: str, username: str, role: str = "user") -> str:
    """Generates a cryptographically signed HMAC-SHA256 JWT token."""
    header = {"alg": "HS256", "typ": "JWT"}
    now = int(time.time())
    payload = {
        "sub": user_id,
        "username": username,
        "role": role,
        "iat": now,
        "exp": now + TOKEN_EXPIRY_SECONDS,
    }

    header_b64 = _base64url_encode(json.dumps(header, separators=(",", ":")).encode("utf-8"))
    payload_b64 = _base64url_encode(json.dumps(payload, separators=(",", ":")).encode("utf-8"))
    signing_input = f"{header_b64}.{payload_b64}".encode("utf-8")

    signature = hmac.new(SECRET_KEY.encode("utf-8"), signing_input, hashlib.sha256).digest()
    signature_b64 = _base64url_encode(signature)

    return f"{header_b64}.{payload_b64}.{signature_b64}"


def verify_access_token(token: str) -> Optional[Dict[str, Any]]:
    """
    Validates cryptographic signature and expiration of JWT token.
    Returns decoded payload if valid, None otherwise.
    """
    if not token or not isinstance(token, str):
        return None

    # Strip Bearer prefix if provided
    if token.startswith("Bearer "):
        token = token[7:].strip()

    parts = token.split(".")
    if len(parts) != 3:
        return None

    header_b64, payload_b64, signature_b64 = parts

    try:
        signing_input = f"{header_b64}.{payload_b64}".encode("utf-8")
        expected_sig = hmac.new(SECRET_KEY.encode("utf-8"), signing_input, hashlib.sha256).digest()
        provided_sig = _base64url_decode(signature_b64)

        if not hmac.compare_digest(expected_sig, provided_sig):
            logger.warning("JWT verification failed: Invalid cryptographic signature.")
            return None

        payload_bytes = _base64url_decode(payload_b64)
        payload = json.loads(payload_bytes.decode("utf-8"))

        if payload.get("exp", 0) < int(time.time()):
            logger.warning("JWT verification failed: Token has expired.")
            return None

        return payload
    except Exception as e:
        logger.warning(f"JWT verification error: {str(e)}")
        return None
