import os
import smtplib
import ssl
import threading
import queue
import asyncio
from typing import Optional, Dict, Any, List
from email.message import EmailMessage

from fastapi import APIRouter, Depends
from dotenv import load_dotenv

from app.security.rbac import require_role

load_dotenv()


router = APIRouter(
    prefix="/api/email",
    tags=["email"],
    dependencies=[Depends(require_role("admin"))],
)


# ======================
# CONFIG (hardcoded sender/receiver for now)
# ======================

# Default violation alert recipients (used if VIOLATION_EMAIL_TO env is unset).
# You can list several addresses here, or set VIOLATION_EMAIL_TO in .env as comma- or semicolon-separated.
DEFAULT_VIOLATION_RECIPIENTS: List[str] = [
    os.environ.get("RECEIVER_EMAIL")
]

GMAIL_ADDRESS: Optional[str] = os.environ.get("EMAIL_ADDRESS")
GMAIL_APP_PASSWORD: Optional[str] = os.environ.get("EMAIL_PASSWORD")
GMAIL_SMTP_SERVER = os.environ.get("SMTP_SERVER")
GMAIL_SMTP_PORT_SSL = os.environ.get("SMTP_PORT")  # using SSL port for Gmail

# Temporary Office365 credentials (from ppe_kit_detector.py 29-32) - not used by this module
# EMAIL_ADDRESS = "eframeAI@outlook.com"
# EMAIL_PASSWORD = "lfmpzajspuopbrrr"
# SMTP_SERVER = "smtp.office365.com"
# SMTP_PORT = 587

def _parse_violation_recipients_from_env() -> List[str]:
    raw = (os.environ.get("VIOLATION_EMAIL_TO") or "").strip()
    if not raw:
        return list(DEFAULT_VIOLATION_RECIPIENTS)
    parts = [p.strip() for p in raw.replace(";", ",").split(",") if p.strip()]
    return parts if parts else list(DEFAULT_VIOLATION_RECIPIENTS)


# All violation emails go to these addresses (one SMTP message, multiple To).
VIOLATION_EMAIL_RECIPIENTS: List[str] = _parse_violation_recipients_from_env()


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
    return bool(GMAIL_ADDRESS and GMAIL_APP_PASSWORD and VIOLATION_EMAIL_RECIPIENTS)


def _send_email_synchronously(payload: Dict[str, Any]) -> None:
    """
    Low-level: send a single email using Gmail SMTP over SSL.
    Runs in the background worker; never called from request/YOLO threads.
    """
    if not _have_valid_email_config():
        return

    subject: str = payload.get("subject", "PPE Violation Detected")
    body: str = payload.get("body", "")
    to_addrs: List[str] = [a for a in (payload.get("to") or VIOLATION_EMAIL_RECIPIENTS) if a]
    if not to_addrs:
        return
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


def _format_camera_line_for_email(
    camera_id: int,
    camera_name: Optional[str],
    zone_name: Optional[str],
) -> str:
    """Match notifications UI: Name · Zone: … (no IP)."""
    name = (camera_name or "").strip() or f"Camera ID {camera_id}"
    if zone_name and str(zone_name).strip():
        return f"{name} · Zone: {zone_name.strip()}"
    return name


def _build_violation_email_payload(
    camera_id: int,
    exception_type: str,
    image_path: str,
    time_occurred: str,
    *,
    camera_name: Optional[str] = None,
    zone_name: Optional[str] = None,
) -> Dict[str, Any]:
    display_name = (camera_name or "").strip() or f"Camera {camera_id}"
    camera_line = _format_camera_line_for_email(camera_id, camera_name, zone_name)
    subject = f"PPE Violation: {exception_type} — {display_name}"
    lines = [
        "PPE violation detected.",
        "",
        f"Camera: {camera_line}",
        "",
        f"Violation   : {exception_type}",
        f"Time        : {time_occurred}",
    ]
    body = "\n".join(lines)

    return {
        "subject": subject,
        "body": body,
        "to": list(VIOLATION_EMAIL_RECIPIENTS),
        "image_path": image_path,
    }


def _enqueue_payload(payload: Dict[str, Any]) -> None:
    _ensure_email_worker()
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


def enqueue_violation_email(
    camera_id: int,
    exception_type: str,
    image_path: str,
    time_occurred: str,
    *,
    camera_name: Optional[str] = None,
    zone_name: Optional[str] = None,
) -> None:
    """
    Non-blocking API used by other modules (e.g. camera_dashboard) to request an email.
    This ONLY enqueues; actual sending is done in a dedicated worker thread.
    Pass camera_name and zone_name from camera.
    """
    global _email_config_warning_logged

    if not _have_valid_email_config():
        # Log once if configuration is missing; avoid spamming logs on every violation.
        if not _email_config_warning_logged:
            print(
                "[email_feature] Email configuration missing. "
                "Set GMAIL_ADDRESS, GMAIL_APP_PASSWORD, and VIOLATION_EMAIL_TO "
                "(comma-separated for multiple recipients) in environment/.env to enable violation email alerts."
            )
            _email_config_warning_logged = True
        return

    payload = _build_violation_email_payload(
        camera_id=camera_id,
        exception_type=exception_type,
        image_path=image_path,
        time_occurred=time_occurred,
        camera_name=camera_name,
        zone_name=zone_name,
    )
    _enqueue_payload(payload)


async def enqueue_violation_email_async(
    camera_id: int,
    exception_type: str,
    image_path: str,
    time_occurred: str,
    *,
    camera_name: Optional[str] = None,
    zone_name: Optional[str] = None,
) -> None:
    """
    Async-friendly wrapper for async callers.
    Enqueue remains non-blocking; SMTP send still runs in background worker thread.
    """
    enqueue_violation_email(
        camera_id=camera_id,
        exception_type=exception_type,
        image_path=image_path,
        time_occurred=time_occurred,
        camera_name=camera_name,
        zone_name=zone_name,
    )
    # Yield once so async pipelines can continue fairly.
    await asyncio.sleep(0)


@router.get("/status")
def email_status() -> dict:
    """Return email config and queue status for diagnostics."""
    out = {
        "configured": _have_valid_email_config(),
        "gmail_address": GMAIL_ADDRESS,
        "violation_email_to": ", ".join(VIOLATION_EMAIL_RECIPIENTS),
        "violation_email_recipients": VIOLATION_EMAIL_RECIPIENTS,
        "queue_size": _email_queue.qsize(),
        "queue_maxsize": EMAIL_QUEUE_MAXSIZE,
        "worker_started": _email_worker_started,
    }
    return out

