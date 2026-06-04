"""
DNSE Auto-Authentication — OTP via Gmail IMAP.

Flow:
1. Call send_otp() → DNSE sends email
2. Wait & poll Gmail IMAP for latest DNSE OTP email
3. Extract OTP code from email body
4. Call verify_otp(code) → get trading token

Requires .env:
  DNSE_OTP_EMAIL=your_gmail@gmail.com
  DNSE_OTP_APP_PASSWORD=xxxx xxxx xxxx xxxx  (Gmail App Password)
"""

import imaplib
import email
import re
import time
from datetime import datetime, timedelta, timezone

from config import Config
from dnse import DnseClient

VN_TZ = timezone(timedelta(hours=7))

IMAP_HOST = "imap.gmail.com"
IMAP_PORT = 993
OTP_WAIT_TIMEOUT = 90  # seconds
OTP_POLL_INTERVAL = 5  # seconds
DNSE_SENDER = "noreply@mail.dnse.com.vn"


def _fetch_otp_from_gmail(email_addr: str, app_password: str,
                          since_time: datetime) -> str | None:
    """Connect to Gmail IMAP and find latest DNSE OTP email."""
    mail = imaplib.IMAP4_SSL(IMAP_HOST, IMAP_PORT)
    try:
        mail.login(email_addr, app_password)
        mail.select("INBOX")

        # Search for emails from DNSE noreply
        date_str = since_time.strftime("%d-%b-%Y")
        _, msg_ids = mail.search(None, f'(SINCE "{date_str}" FROM "{DNSE_SENDER}")')

        if not msg_ids[0]:
            return None

        ids = msg_ids[0].split()
        # Check last 3 emails (most recent first)
        for msg_id in reversed(ids[-3:]):
            _, msg_data = mail.fetch(msg_id, "(RFC822)")
            raw = msg_data[0][1]
            msg = email.message_from_bytes(raw)

            # Check email time is after our OTP request
            date_header = msg.get("Date", "")
            try:
                msg_time = email.utils.parsedate_to_datetime(date_header)
                # Allow 10s tolerance before send time
                if msg_time < since_time - timedelta(seconds=10):
                    continue
            except Exception:
                pass

            # Extract body
            body = _get_email_body(msg)
            if not body:
                continue

            # Must be OTP email (not invoice or other)
            if "OTP" not in body and "otp" not in body.lower():
                continue

            # Extract OTP (6 digits on its own line)
            otp = _extract_otp(body)
            if otp:
                return otp

        return None
    finally:
        try:
            mail.logout()
        except Exception:
            pass


def _get_email_body(msg) -> str:
    """Extract text body from email message."""
    if msg.is_multipart():
        for part in msg.walk():
            ctype = part.get_content_type()
            if ctype == "text/plain":
                payload = part.get_payload(decode=True)
                if payload:
                    charset = part.get_content_charset() or "utf-8"
                    return payload.decode(charset, errors="replace")
            elif ctype == "text/html":
                payload = part.get_payload(decode=True)
                if payload:
                    charset = part.get_content_charset() or "utf-8"
                    html = payload.decode(charset, errors="replace")
                    # Simple HTML to text
                    text = re.sub(r'<[^>]+>', ' ', html)
                    return text
    else:
        payload = msg.get_payload(decode=True)
        if payload:
            charset = msg.get_content_charset() or "utf-8"
            return payload.decode(charset, errors="replace")
    return ""


def _extract_otp(text: str) -> str | None:
    """Extract 6-digit OTP code from DNSE email text.

    DNSE format: OTP code appears on its own line as exactly 6 digits.
    Example: '...của Quý khách là:\\n\\n562587\\n\\nMã OTP có hiệu lực...'
    """
    # Pattern 1: 6 digits on its own line (most reliable for DNSE)
    match = re.search(r'(?:^|\n)\s*(\d{6})\s*(?:\n|$)', text)
    if match:
        return match.group(1)

    # Pattern 2: after "là:" or "is:"
    match = re.search(r'(?:là|is)[:\s]*(\d{6})', text)
    if match:
        return match.group(1)

    # Pattern 3: any 6-digit sequence near OTP keyword
    match = re.search(r'OTP.*?(\d{6})', text, re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1)

    return None


def auto_authenticate(client: DnseClient, max_retries: int = 2) -> str:
    """Full auto-auth flow: send OTP → read from Gmail → verify.

    Args:
        client: Initialized DnseClient (with api_key/secret).
        max_retries: Number of OTP send retries.

    Returns:
        Trading token string.

    Raises:
        RuntimeError if OTP cannot be retrieved or verified.
    """
    otp_email = Config.DNSE_OTP_EMAIL
    app_password = Config.DNSE_OTP_APP_PASSWORD

    if not otp_email or not app_password:
        raise RuntimeError(
            "DNSE_OTP_EMAIL and DNSE_OTP_APP_PASSWORD must be set in .env"
        )

    for attempt in range(max_retries):
        # Record time before sending OTP (UTC for comparison with email Date header)
        before_send = datetime.now(timezone.utc)
        print(f"[AUTH] Sending OTP to {otp_email}...")

        try:
            client.registration.send_otp()
        except Exception as e:
            print(f"[AUTH] send_otp failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(5)
                continue
            raise RuntimeError(f"Failed to send OTP: {e}")

        print(f"[AUTH] OTP sent. Waiting for email...")
        # Wait a bit for email to arrive before first poll
        time.sleep(8)

        # Poll Gmail for OTP
        deadline = time.time() + OTP_WAIT_TIMEOUT
        otp_code = None

        while time.time() < deadline:
            time.sleep(OTP_POLL_INTERVAL)
            try:
                otp_code = _fetch_otp_from_gmail(otp_email, app_password, before_send)
            except Exception as e:
                print(f"[AUTH] IMAP error: {e}")
                continue

            if otp_code:
                break

        if not otp_code:
            print(f"[AUTH] OTP not found in email (attempt {attempt + 1}/{max_retries})")
            if attempt < max_retries - 1:
                continue
            raise RuntimeError("Could not retrieve OTP from email within timeout")

        # Verify OTP
        print(f"[AUTH] Got OTP: {otp_code}. Verifying...")
        try:
            token = client.registration.verify_otp(otp_code)
            print(f"[AUTH] Trading token obtained successfully.")
            return token
        except Exception as e:
            print(f"[AUTH] verify_otp failed: {e}")
            if attempt < max_retries - 1:
                continue
            raise RuntimeError(f"OTP verification failed: {e}")

    raise RuntimeError("Authentication failed after all retries")


def create_authenticated_client() -> DnseClient:
    """Create a DnseClient with automatic OTP authentication.

    Returns ready-to-trade client with trading token set.
    Token valid for 8 hours (full trading day).
    """
    client = DnseClient(
        api_key=Config.DNSE_API_KEY,
        api_secret=Config.DNSE_API_SECRET,
    )
    auto_authenticate(client)
    return client


def ensure_token_valid(client: DnseClient, auth_time: datetime) -> bool:
    """Check if trading token is still valid (8h lifetime). Re-auth if needed.

    Args:
        client: DnseClient instance
        auth_time: datetime when token was obtained

    Returns:
        True if token is valid or re-auth succeeded.
    """
    elapsed = (datetime.now(timezone.utc) - auth_time).total_seconds()
    if elapsed < 7 * 3600:  # refresh at 7h to be safe
        return True

    print("[AUTH] Token expiring soon, re-authenticating...")
    try:
        auto_authenticate(client)
        return True
    except Exception as e:
        print(f"[AUTH] Re-auth failed: {e}")
        return False
