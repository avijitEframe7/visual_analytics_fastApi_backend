import os
import smtplib
import ssl
import threading
import queue
from typing import Optional, Dict, Any, List
from email.message import EmailMessage

from fastapi import APIRouter


router = APIRouter(prefix="/api/email", tags=["email"])


# ======================
# CONFIG (hardcoded sender/receiver for now)
# ======================

# Receiver
RECEIVER_EMAIL = "avijit.eframe@gmail.com"

GMAIL_ADDRESS: Optional[str] = "eframeinterns@gmail.com"
GMAIL_APP_PASSWORD: Optional[str] = "ibfx koos skrd rinb"
GMAIL_SMTP_SERVER = "smtp.gmail.com"
GMAIL_SMTP_PORT_SSL = 465  # using SSL port for Gmail

# Temporary Office365 credentials (from ppe_kit_detector.py 29-32) - not used by this module
EMAIL_ADDRESS = "eframeAI@outlook.com"
EMAIL_PASSWORD = "lfmpzajspuopbrrr"
SMTP_SERVER = "smtp.office365.com"
SMTP_PORT = 587

# For now, send all violation emails to this receiver
VIOLATION_EMAIL_TO: Optional[str] = RECEIVER_EMAIL


# ======================
# EMAIL QUEUE & WORKER
# ======================

EMAIL_QUEUE_MAXSIZE = 128
_email_queue: "queue.Queue[Dict[str, Any]]" = queue.Queue(maxsize=EMAIL_QUEUE_MAXSIZE)
_email_worker_started = False
_email_worker_lock = threading.Lock()
_email_config_warning_logged = False


def _have_valid_email_config() -> bool:
    """
    True if we have enough configuration to try sending email.
    """
    return bool(GMAIL_ADDRESS and GMAIL_APP_PASSWORD and VIOLATION_EMAIL_TO)


def _send_email_synchronously(payload: Dict[str, Any]) -> None:
    """
    Low-level: send a single email using Gmail SMTP over SSL.
    Runs in the background worker; never called from request/YOLO threads.
    """
    if not _have_valid_email_config():
        return

    subject: str = payload.get("subject", "PPE Violation Detected")
    body: str = payload.get("body", "")
    to_addrs: List[str] = payload.get("to", [VIOLATION_EMAIL_TO]) or [VIOLATION_EMAIL_TO]  # type: ignore[list-item]
    image_path: Optional[str] = payload.get("image_path")

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = GMAIL_ADDRESS
    msg["To"] = ", ".join(to_addrs)
    msg.set_content(body)

    # Attach image if available
    if image_path and os.path.isfile(image_path):
        try:
            with open(image_path, "rb") as f:
                data = f.read()
            msg.add_attachment(
                data,
                maintype="image",
                subtype="jpeg",
                filename=os.path.basename(image_path),
            )
        except Exception as e:
            # Log to stdout; do not raise so other emails can proceed
            print(f"[email_feature] Failed to attach image '{image_path}': {e}")

    context = ssl.create_default_context()
    try:
        with smtplib.SMTP_SSL(GMAIL_SMTP_SERVER, GMAIL_SMTP_PORT_SSL, context=context, timeout=10) as server:
            server.login(GMAIL_ADDRESS, GMAIL_APP_PASSWORD)  # type: ignore[arg-type]
            server.send_message(msg)
        print(f"[email_feature] Sent email to {to_addrs}: subject='{subject}'")
    except Exception as e:
        print(f"[email_feature] Error sending email: {e}")


def _email_worker() -> None:
    """
    Background thread that consumes the email queue and sends emails.
    """
    print("[email_feature] Email worker started")
    while True:
        try:
            payload = _email_queue.get()
        except Exception:
            continue
        if payload is None:
            # Reserved for future graceful shutdown; not used now.
            break
        try:
            _send_email_synchronously(payload)
        except Exception as e:
            print(f"[email_feature] Unexpected error in email worker: {e}")


def _ensure_email_worker() -> None:
    """
    Start the background worker once (thread-safe).
    """
    global _email_worker_started
    if _email_worker_started:
        return
    with _email_worker_lock:
        if _email_worker_started:
            return
        t = threading.Thread(target=_email_worker, daemon=True, name="Violation-Email-Worker")
        t.start()
        _email_worker_started = True


def enqueue_violation_email(
    camera_id: int,
    exception_type: str,
    image_path: str,
    time_occurred: str,
) -> None:
    """
    Non-blocking API used by other modules (e.g. camera_dashboard) to request an email.
    This ONLY enqueues; actual sending is done in a dedicated worker thread.
    """
    global _email_config_warning_logged

    if not _have_valid_email_config():
        # Log once if configuration is missing; avoid spamming logs on every violation.
        if not _email_config_warning_logged:
            print(
                "[email_feature] Email configuration missing. "
                "Set GMAIL_ADDRESS, GMAIL_APP_PASSWORD, and optionally VIOLATION_EMAIL_TO "
                "in environment/.env to enable violation email alerts."
            )
            _email_config_warning_logged = True
        return

    _ensure_email_worker()

    subject = f"PPE Violation: {exception_type} (Camera {camera_id})"
    lines = [
        f"PPE violation detected.",
        f"Camera ID   : {camera_id}",
        f"Violation   : {exception_type}",
        f"Time        : {time_occurred}",
    ]
    if image_path:
        lines.append(f"Image path  : {image_path}")
    body = "\n".join(lines)

    payload = {
        "subject": subject,
        "body": body,
        "to": [VIOLATION_EMAIL_TO],
        "image_path": image_path,
    }

    try:
        _email_queue.put_nowait(payload)
    except queue.Full:
        # Drop the oldest email to keep queue fresh and avoid blocking
        try:
            _email_queue.get_nowait()
            _email_queue.put_nowait(payload)
            print("[email_feature] Email queue full, dropped oldest item")
        except queue.Empty:
            pass


@router.get("/status")
def email_status() -> dict:
    """Return email config and queue status for diagnostics."""
    out = {
        "configured": _have_valid_email_config(),
        "gmail_address": GMAIL_ADDRESS,
        "violation_email_to": VIOLATION_EMAIL_TO,
        "queue_size": _email_queue.qsize(),
        "queue_maxsize": EMAIL_QUEUE_MAXSIZE,
        "worker_started": _email_worker_started,
    }
    return out

