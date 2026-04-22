import asyncio
import json
import logging
import math
import os
import queue
import re
import sys
import threading
import time
from datetime import datetime
from typing import Any, List, Optional, Set, Tuple
from urllib.parse import quote, unquote

import cv2
import numpy as np
import torch
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy import text
from ultralytics import YOLO

from app.database.database import SessionLocal
from app.routes.camera_config import get_camera_config, get_rtsp_urls
from app.routes.email_feature import enqueue_violation_email
from app.security.rbac import (
    decode_access_token,
    get_user_allowed_page_keys,
    require_permission,
)

# Add TensorRT DLL search paths before any TensorRT/Ultralytics use (fixes nvinfer_10.dll not found)
_script_dir = os.path.dirname(os.path.abspath(__file__))
_pf = os.environ.get("ProgramFiles", "C:\\Program Files")
_trt_paths = [
    os.path.join(_script_dir, ".venv", "Lib", "site-packages", "tensorrt.libs"),
    os.path.join(_pf, "NVIDIA GPU Computing Toolkit", "TensorRT", "bin"),
    os.path.join(_pf, "NVIDIA GPU Computing Toolkit", "TensorRT", "lib"),
    os.path.join(_pf, "TensorRT-10.11.0.33", "lib"),  # nvinfer_10.dll is here
    os.path.join(_pf, "TensorRT-10.11.0.33", "bin"),
]
for _p in _trt_paths:
    if os.path.isdir(_p) and _p not in os.environ.get("PATH", ""):
        os.environ["PATH"] = _p + os.pathsep + os.environ.get("PATH", "")


# -----------------------------------------------------------------------------
# Config: streams, inference tuning, reconnect
# -----------------------------------------------------------------------------
RTSP_URLS: List[str] = []

# Live RTSP: process every Nth frame (1 = all frames, 2 = every other frame, etc.). Used by rtsp_reader only.
FRAME_SKIP = 1
# File upload / video analysis: 0 = do not skip any frame (run model on every decoded frame). N>=2 = only every Nth.
FILE_ANALYSIS_FRAME_SKIP = 0
RESIZE = (640, 460)
BATCH_SIZE = 4
QUEUE_SIZE = 50
BATCH_TIMEOUT = 0.5          # (s) process partial batch after this (multi-camera)
SINGLE_STREAM_BATCH_TIMEOUT = 0.05  # (s) much shorter so single-camera doesn't wait 0.5s → ~2 FPS
MAX_RECONNECT_ATTEMPTS = 5
RECONNECT_DELAY = 2

# -----------------------------------------------------------------------------
# Pipeline state: per-camera queues, locks, stats, filters
# -----------------------------------------------------------------------------
stats_lock = threading.Lock()
pipeline_input_queues: List[queue.Queue] = []
pipeline_output_queues: List[queue.Queue] = []
pipeline_display_queues: List[queue.Queue] = []   # consumed by live_detection_feed API / WebSocket
pipeline_imshow_queues: List[queue.Queue] = []    # consumed only by display_worker (cv2.imshow); separate so API is unaffected

# Performance stats
performance_stats = {
    'frames_processed': 0,
    'frames_dropped': 0,
    'batches_processed': 0,
    'fps': 0.0,
    'last_update': time.time()
}

# Class filter: None or empty = show all; else only these class names (lowercase) are shown
selected_class_names: Optional[Set[str]] = None

# User selects PPE type (e.g. Helmet) = show VIOLATIONS (who is NOT wearing it). Map frontend id -> model violation class (normalized).
# Model classes from training: Helmet, Safety_Vest, Safety_goggles, Safety_shoes, NO_helmet, NO_Vest, NO_goggles, NO_safetyshoes
USER_SELECTION_TO_VIOLATION: dict = {
    "helmet": ["no_helmet"],
    "safety_vest": ["no_vest"],
    "goggles": ["no_goggles"],
    "safety_goggles": ["no_goggles"],
    "shoes": ["no_safetyshoes"],
    "safety_shoes": ["no_safetyshoes"],
}

# Exception log: violations to insert into exception_logs (notification feed)
# Model training names (normalized) vs exception_type.exception_name — map below when they differ.
VIOLATION_CLASSES_FOR_LOG = frozenset(
    {"no_helmet", "no_vest", "no_goggles", "no_safetyshoes", "no_gloves"}
)

# Rules engine: violations only if a person is detected and the violation associates with a person box.
# Person boxes below this confidence are ignored when gating violations.
PERSON_MIN_CONF = 0.5
# Minimum model score to treat a box as a violation (shoes/gloves: lower so faint detections still get spatial checks).
VIOLATION_CLASS_MIN_CONF: dict[str, float] = {
    "no_helmet": 0.45,
    "no_vest": 0.45,
    "no_goggles": 0.38,
    "no_safetyshoes": 0.28,
    "no_gloves": 0.18,
}
DEFAULT_VIOLATION_MIN_CONF = 0.40
# Min association score vs some person box: max(intersection(V,P)/area(V), center(V) in P ? 1 : 0).
# Lower for feet/hands where boxes sit on edges of the person bbox.
VIOLATION_ASSOCIATION_MIN: dict[str, float] = {
    "no_helmet": 0.45,
    "no_vest": 0.45,
    "no_goggles": 0.28,
    "no_safetyshoes": 0.18,
    "no_gloves": 0.10,
}
DEFAULT_VIOLATION_ASSOCIATION_MIN = 0.28

# no_vest: stricter than max(overlap, center) — torso FPs often sit beside clutter with high "overlap" vs a loose person box.
# Require bbox center inside a person box AND at least this fraction of the vest box area overlapping that person.
NO_VEST_STRICT_CENTER_IN_PERSON = True
NO_VEST_MIN_AREA_OVERLAP_WITH_PERSON = 0.55

# YOLO emits short names; employeeinfo.exception_type uses longer labels (see SSMS seed rows).
MODEL_NORMALIZED_TO_DB_EXCEPTION_NAME: dict[str, str] = {
    "no_vest": "no_safety_vest",
    "no_safetyshoes": "no_safety_shoes",
}
EXCEPTION_LOG_THROTTLE_SECONDS = 180
EXCEPTION_LOG_QUEUE_SIZE = 32
exception_log_queue: queue.Queue = queue.Queue(maxsize=EXCEPTION_LOG_QUEUE_SIZE)
_last_exception_log_time: dict = {}  # (cam_id, exception_type) -> time.time()
_exception_log_time_lock = threading.Lock()

# Realtime detection JSON queue: per-frame detection summaries for external consumers
DETECTION_JSON_QUEUE_SIZE = 256
detection_json_queue: queue.Queue = queue.Queue(maxsize=DETECTION_JSON_QUEUE_SIZE)

# Detection frame storage: save images with detections to media/detection_frames/
DETECTION_FRAMES_QUEUE_SIZE = 64
DETECTION_FRAME_THROTTLE_SECONDS = 5
detection_frames_queue: queue.Queue = queue.Queue(maxsize=DETECTION_FRAMES_QUEUE_SIZE)
_last_detection_frame_time: dict = {}
_detection_frame_time_lock = threading.Lock()

# Media dirs: <backend>/media/{exception_logs,detection_frames,logs.txt}
MEDIA_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "media"))
EXCEPTION_LOGS_DIR = os.path.abspath(os.path.join(MEDIA_ROOT, "exception_logs"))
DETECTION_FRAMES_DIR = os.path.abspath(os.path.join(MEDIA_ROOT, "detection_frames"))
DETECTION_LOG_FILE = os.path.abspath(os.path.join(MEDIA_ROOT, "logs.txt"))
EXCEPTION_PIPELINE_LOG_PATH = os.path.abspath(os.path.join(MEDIA_ROOT, "exception_pipeline.log"))

# Uploaded file analysis (demo2/3/4): stored under media/uploads
UPLOAD_FOLDER = os.path.abspath(os.path.join(MEDIA_ROOT, "uploads"))
ALLOWED_UPLOAD_EXTENSIONS = frozenset({"mp4", "avi", "mov", "mkv", "jpg", "jpeg", "png"})

_exception_pipeline_logger: Optional[logging.Logger] = None


def _log_exception_pipeline(msg: str) -> None:
    """Print to stdout and append to media/exception_pipeline.log (survives headless / service runs)."""
    global _exception_pipeline_logger
    print(f"[camera_dashboard] {msg}")
    try:
        os.makedirs(MEDIA_ROOT, exist_ok=True)
        if _exception_pipeline_logger is None:
            lg = logging.getLogger("camera_dashboard.exception_pipeline")
            lg.setLevel(logging.INFO)
            lg.handlers.clear()
            fh = logging.FileHandler(EXCEPTION_PIPELINE_LOG_PATH, encoding="utf-8")
            fh.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
            lg.addHandler(fh)
            lg.propagate = False
            _exception_pipeline_logger = lg
        _exception_pipeline_logger.info(msg)
    except Exception as e:
        print(f"[camera_dashboard] exception_log: failed to write exception_pipeline.log: {e}")


def _raw_class_name_from_names(names: Any, cls_idx: int) -> str:
    """Ultralytics may expose class names as dict (id -> str) or list; avoid .get on lists."""
    ci = int(cls_idx)
    if isinstance(names, dict):
        return str(names.get(ci, str(ci)))
    try:
        if 0 <= ci < len(names):
            return str(names[ci])
    except (TypeError, IndexError, KeyError):
        pass
    return str(ci)


def _normalize_exception_name_for_db(s: str) -> str:
    """Match model output (e.g. no_helmet) to exception_type.exception_name (may be NO_helmet, 'No Helmet', etc.)."""
    x = (s or "").strip().lower().replace(" ", "_").replace("-", "_")
    while "__" in x:
        x = x.replace("__", "_")
    return x.strip("_")


def _bbox_area_xyxy(xyxy) -> float:
    x1, y1, x2, y2 = float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3])
    w = max(0.0, x2 - x1)
    h = max(0.0, y2 - y1)
    return w * h


def _intersection_area_xyxy(a, b) -> float:
    x1 = max(float(a[0]), float(b[0]))
    y1 = max(float(a[1]), float(b[1]))
    x2 = min(float(a[2]), float(b[2]))
    y2 = min(float(a[3]), float(b[3]))
    if x2 <= x1 or y2 <= y1:
        return 0.0
    return (x2 - x1) * (y2 - y1)


def _association_score_v_person(v_xyxy, p_xyxy) -> float:
    """How well a violation box sits on a person: overlap fraction and/or center inside person."""
    av = _bbox_area_xyxy(v_xyxy)
    inter = _intersection_area_xyxy(v_xyxy, p_xyxy)
    frac = (inter / av) if av > 1e-6 else 0.0
    cx = (float(v_xyxy[0]) + float(v_xyxy[2])) * 0.5
    cy = (float(v_xyxy[1]) + float(v_xyxy[3])) * 0.5
    px1, py1, px2, py2 = float(p_xyxy[0]), float(p_xyxy[1]), float(p_xyxy[2]), float(p_xyxy[3])
    center_in = 1.0 if (px1 <= cx <= px2 and py1 <= cy <= py2) else 0.0
    return max(frac, center_in)


def _best_violation_person_association(v_xyxy, person_boxes: List[np.ndarray]) -> float:
    best = 0.0
    for p in person_boxes:
        best = max(best, _association_score_v_person(v_xyxy, p))
    return best


def _bbox_center_xy(v_xyxy) -> Tuple[float, float]:
    return (
        (float(v_xyxy[0]) + float(v_xyxy[2])) * 0.5,
        (float(v_xyxy[1]) + float(v_xyxy[3])) * 0.5,
    )


def _point_inside_xyxy(px: float, py: float, xyxy) -> bool:
    x1, y1, x2, y2 = float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3])
    return x1 <= px <= x2 and y1 <= py <= y2


def _no_vest_strictly_on_person(v_xyxy, person_boxes: List[np.ndarray]) -> bool:
    """
    Vest violations must sit on a person: center of the vest box inside a person bbox and
    most of the vest area overlapping that same person (reduces FPs beside equipment/walls).
    """
    av = _bbox_area_xyxy(v_xyxy)
    if av <= 1e-6:
        return False
    cx, cy = _bbox_center_xy(v_xyxy)
    need_frac = NO_VEST_MIN_AREA_OVERLAP_WITH_PERSON
    for p in person_boxes:
        if not _point_inside_xyxy(cx, cy, p):
            continue
        inter = _intersection_area_xyxy(v_xyxy, p)
        if inter / av >= need_frac:
            return True
    return False


def _person_boxes_from_frame(
    xyxy_np: np.ndarray,
    cls_np: np.ndarray,
    conf_np: np.ndarray,
    names: Any,
) -> List[np.ndarray]:
    """Collect person boxes above PERSON_MIN_CONF (normalized class name == person)."""
    out: List[np.ndarray] = []
    for i in range(len(cls_np)):
        if float(conf_np[i]) < PERSON_MIN_CONF:
            continue
        cls_idx = int(cls_np[i])
        raw = _raw_class_name_from_names(names, cls_idx)
        norm = _normalize_exception_name_for_db(raw)
        if norm != "person":
            continue
        out.append(xyxy_np[i])
    return out


def violation_passes_person_rules(
    norm_name: str,
    score: float,
    v_xyxy: np.ndarray,
    person_boxes: List[np.ndarray],
) -> bool:
    """
    True if this box should count as an actionable PPE violation:
    class is a violation, score meets per-class floor, at least one person exists,
    and the box associates with a person bbox (overlap or center-in-person).
    """
    if norm_name not in VIOLATION_CLASSES_FOR_LOG:
        return False
    min_conf = VIOLATION_CLASS_MIN_CONF.get(norm_name, DEFAULT_VIOLATION_MIN_CONF)
    if float(score) < min_conf:
        return False
    if not person_boxes:
        return False
    if norm_name == "no_vest" and NO_VEST_STRICT_CENTER_IN_PERSON:
        return _no_vest_strictly_on_person(v_xyxy, person_boxes)
    assoc_min = VIOLATION_ASSOCIATION_MIN.get(norm_name, DEFAULT_VIOLATION_ASSOCIATION_MIN)
    return _best_violation_person_association(v_xyxy, person_boxes) >= assoc_min


def _resolve_exception_type_id(db, exception_type: str) -> Optional[int]:
    """
    Resolve FK for exception_logs. Exact equality on exception_name often fails when the DB
    uses different casing or spacing than the normalized YOLO class name.
    Also maps model names to exception_type rows (e.g. no_vest -> no_safety_vest).
    """
    norm = _normalize_exception_name_for_db(exception_type)
    db_label = MODEL_NORMALIZED_TO_DB_EXCEPTION_NAME.get(norm, norm)
    sql_norm = _normalize_exception_name_for_db(db_label)
    row = db.execute(
        text(
            """
            SELECT et.exception_type_id
            FROM exception_type et
            WHERE LOWER(REPLACE(REPLACE(TRIM(et.exception_name), ' ', '_'), '-', '_')) = :norm
            LIMIT 1
            """
        ),
        {"norm": sql_norm},
    ).scalar()
    if row is not None:
        return int(row)
    rows = db.execute(
        text(
            "SELECT exception_type_id, exception_name FROM exception_type ORDER BY exception_type_id"
        )
    ).fetchall()
    _log_exception_pipeline(
        f"exception_log: no exception_type_id for model class {exception_type!r} "
        f"(normalized={norm!r}, db_lookup={sql_norm!r}). Check exception_type.exception_name. "
        f"Current rows: {rows}"
    )
    return None


# Counters for debugging (YOLO worker)
_debug_frames_with_detections = 0
_debug_batches_processed = 0
_debug_lock = threading.Lock()

# Stop signal for live detection pipeline
pipeline_stop_event = threading.Event()

# File upload / recorded video analysis (demo2, demo3, demo4)
video_processing_active = False
current_processing_type: Optional[str] = None
current_processing_video_path: Optional[str] = None
video_processing_stop_requested = False
file_analysis_selected_classes: List[str] = []

# Ordered camera ids for current pipeline (index = display queue index)
current_camera_ids: List[str] = []


def _pipeline_index_to_db_camera_id(cam_index) -> int:
    """
    RTSP/YOLO threads tag frames with a stream index (0..n-1). camera uses the real PK
    (see current_camera_ids from start_live_detection). Storing the index as camera_id breaks
    JOINs to camera; map index -> actual camera_id for exception_logs and email.
    """
    global current_camera_ids
    try:
        idx = int(cam_index)
    except (TypeError, ValueError):
        idx = 0
    if 0 <= idx < len(current_camera_ids):
        raw = str(current_camera_ids[idx]).strip()
        if raw.isdigit():
            return int(raw)
    return idx


def _db_str_cell(val) -> Optional[str]:
    if val is None:
        return None
    if isinstance(val, bytes):
        try:
            return val.decode("utf-8").strip() or None
        except UnicodeDecodeError:
            return None
    s = str(val).strip()
    return s or None


def _camera_name_zone_from_db(db, camera_id: int) -> Tuple[Optional[str], Optional[str]]:
    """Load camera_name and zone_name from camera for emails (same source as SSMS)."""
    row = db.execute(
        text(
            """
            SELECT
                camera_name,
                zone_name
            FROM camera
            WHERE camera_id = :cid
            LIMIT 1
            """
        ),
        {"cid": camera_id},
    ).mappings().first()
    if not row:
        return None, None
    m = {k.lower(): v for k, v in dict(row).items()}
    return _db_str_cell(m.get("camera_name")), _db_str_cell(m.get("zone_name"))


def _ensure_detection_log_file():
    """
    Ensure that the plain-text detections log file exists so it is visible
    even before the first detection is written.
    """
    try:
        os.makedirs(MEDIA_ROOT, exist_ok=True)
        if not os.path.exists(DETECTION_LOG_FILE):
            with open(DETECTION_LOG_FILE, "a", encoding="utf-8"):
                pass
    except Exception as e:
        print(f"[camera_dashboard] Failed to ensure detection log file: {e}")

# -----------------------------------------------------------------------------
# YOLO model (GPU; loaded on first /start_live_detection, not on import)
# -----------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "ML_models", "latest_16_1_2026.engine"))
model = None
DEVICE = "cuda:0"


def _tensorrt_python_available() -> bool:
    try:
        import tensorrt  # noqa: F401
        return True
    except ModuleNotFoundError:
        return False


def _resolve_yolo_weights_path(preferred: str) -> str:
    """
    Prefer the configured path. For .engine, require the tensorrt Python package;
    if it is missing, use a same-stem .pt next to the .engine when present.
    """
    preferred = os.path.abspath(preferred)
    if not preferred.lower().endswith(".engine"):
        return preferred
    if _tensorrt_python_available():
        return preferred
    stem, _ = os.path.splitext(preferred)
    pt_path = stem + ".pt"
    if os.path.isfile(pt_path):
        print(
            "[camera_dashboard] TensorRT Python module not found; "
            f"using PyTorch weights: {os.path.basename(pt_path)}"
        )
        return pt_path
    print(
        "[camera_dashboard] ERROR: .engine requires `tensorrt` (pip install tensorrt) "
        f"or a sibling .pt file; missing: {os.path.basename(pt_path)}"
    )
    sys.exit(1)


def _ensure_model_loaded():
    """Load YOLO once when pipeline starts; requires GPU."""
    global model
    if model is not None:
        return
    if not torch.cuda.is_available():
        print("[camera_dashboard] ERROR: CUDA GPU required. Exiting.")
        sys.exit(1)
    torch.cuda.set_device(0)
    gpu_name = torch.cuda.get_device_name(0)
    weights_path = _resolve_yolo_weights_path(MODEL_PATH)
    model = YOLO(weights_path)
    if weights_path.lower().endswith(".pt"):
        model.to(DEVICE)
        model.model.half()
    if weights_path.lower().endswith(".engine"):
        try:
            model(np.zeros((640, 640, 3), dtype=np.uint8), device=DEVICE, verbose=False)
        except Exception as e:
            if "nvinfer" in str(e) or "Could not find" in str(e):
                print("[camera_dashboard] TensorRT DLL not found. Add TensorRT bin to PATH or use .pt model.")
                sys.exit(1)
            raise
    print(f"[camera_dashboard] GPU: {gpu_name} | Model: {os.path.basename(weights_path)}")


# -----------------------------------------------------------------------------
# File upload analysis: helpers + MJPEG generators (demo2 / demo3 / demo4)
# -----------------------------------------------------------------------------
def _allowed_upload_file(filename: Optional[str]) -> bool:
    if not filename or "." not in filename:
        return False
    ext = filename.rsplit(".", 1)[1].lower()
    return ext in ALLOWED_UPLOAD_EXTENSIONS


def _secure_upload_filename(filename: str) -> str:
    base = os.path.basename(filename).replace("..", "")
    cleaned = re.sub(r"[^a-zA-Z0-9._-]", "_", base)
    return cleaned or "upload.bin"


def _safe_path_under_upload(video_path: str) -> Optional[str]:
    """Reject path traversal; only files under UPLOAD_FOLDER."""
    if not video_path:
        return None
    try:
        decoded = unquote(video_path)
        abs_path = os.path.abspath(decoded)
        root = os.path.abspath(UPLOAD_FOLDER)
        if not abs_path.startswith(root + os.sep) and abs_path != root:
            return None
        if not os.path.isfile(abs_path):
            return None
        return abs_path
    except Exception:
        return None


# File-analysis MJPEG: same JPEG quality as live_detection_feed (default quality=82).
LIVE_MJPEG_JPEG_QUALITY = 82

def _file_analysis_pacing_max_fps() -> float:
    """
    Wall-clock cap for how fast the file-analysis MJPEG can advance (independent of video metadata).
    Lower = calmer on-screen (e.g. 6–10 feels like "normal" review speed). Range 1–30.

    Set in environment (optional): FILE_ANALYSIS_PACING_FPS_MAX=8
    """
    try:
        raw = (os.environ.get("FILE_ANALYSIS_PACING_FPS_MAX", "") or "").strip()
        v = float(raw) if raw else 8.0
    except ValueError:
        v = 8.0
    return max(1.0, min(30.0, v))


def _file_analysis_frame_period_sec(cap: cv2.VideoCapture) -> float:
    """
    Target seconds per *decoded* frame: min(video metadata FPS, _file_analysis_pacing_max_fps()).
    If the clip still “flies”, lower FILE_ANALYSIS_PACING_FPS_MAX (e.g. 5–8).
    """
    cap_on = _file_analysis_pacing_max_fps()
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if math.isnan(fps) or fps <= 1.0 or fps > 120.0:
        fps = 0.0
    if fps <= 0.0:
        fps = cap_on
    # Slower/unknown clips: at least 3 FPS so we do not hang on a single frame.
    fps = min(cap_on, max(3.0, fps))
    return 1.0 / fps


def _file_analysis_wait_frame_slot(period: float, slot: List[float]) -> None:
    """
    Block until the next wall-clock frame slot, then schedule the next one.
    Called once per cap.read() *before* processing so the MJPEG `yield` cannot
    cause the next frame to start as soon as the client drains (which looked “too fast”).

    slot[0] = next start time (perf_counter), initialized on first call.
    """
    if not slot:
        slot.append(time.perf_counter())
    deadline = slot[0]
    now = time.perf_counter()
    if now < deadline:
        time.sleep(deadline - now)
    slot[0] = deadline + period
    if time.perf_counter() > slot[0]:
        slot[0] = time.perf_counter() + period


def _file_analysis_should_skip_frame(frame_n: int) -> bool:
    """True to skip this frame (no YOLO / no output). FILE_ANALYSIS_FRAME_SKIP 0 = never skip."""
    n = FILE_ANALYSIS_FRAME_SKIP
    if n <= 0:
        return False
    return (frame_n % n) != 0


def _file_analysis_resize_for_inference(bgr: np.ndarray) -> np.ndarray:
    """Match rtsp_reader: fixed size before YOLO (same as live stream)."""
    if bgr is None or bgr.size == 0:
        return bgr
    if bgr.shape[:2] != RESIZE[::-1]:
        return cv2.resize(bgr, RESIZE, interpolation=cv2.INTER_LINEAR)
    return bgr


def _file_analysis_infer(
    bgr: np.ndarray,
    *,
    conf: float = 0.25,
    iou: float = 0.45,
    classes: Optional[List[int]] = None,
):
    """
    Same call style as yolo_batch_worker: model([frame], device=DEVICE, ...).
    Returns first result (with .boxes) for _annotate_frame, or None on failure.
    """
    _ensure_model_loaded()
    try:
        with torch.inference_mode():
            kwargs: dict = {
                "device": DEVICE,
                "stream": False,
                "verbose": False,
                "conf": conf,
                "iou": iou,
            }
            if classes is not None and len(classes) > 0:
                kwargs["classes"] = classes
            res = model([bgr], **kwargs)
            return res[0] if res else None
    except Exception as e:
        print(f"[file_analysis] YOLO infer error: {e}")
        return None


def generate_processed_frames2(video_path: str):
    """
    MJPEG from file: same RESIZE and YOLO call style as live RTSP; wall-clock throttled; optional
    FILE_ANALYSIS_FRAME_SKIP (0 = all frames).
    """
    global video_processing_stop_requested, model
    global video_processing_active, current_processing_video_path, current_processing_type
    _ensure_model_loaded()
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        video_processing_active = False
        current_processing_video_path = None
        current_processing_type = None
        return
    period = _file_analysis_frame_period_sec(cap)
    wall_clock: List[float] = []
    frame_counter = 0
    try:
        while True:
            if video_processing_stop_requested:
                break
            _file_analysis_wait_frame_slot(period, wall_clock)
            success, img = cap.read()
            if not success:
                break
            frame_counter += 1
            if _file_analysis_should_skip_frame(frame_counter):
                continue
            if img is None or img.size == 0:
                continue
            img = _file_analysis_resize_for_inference(img)
            curr_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            r = _file_analysis_infer(img, conf=0.25, iou=0.45, classes=None)
            if r is None:
                continue
            try:
                annotated_img = _annotate_frame(img, r, filter_by_selected=False)
            except Exception as results_error:
                print(f"[file_analysis] annotate error frame {frame_counter}: {results_error}")
                continue
            boxes = r.boxes
            if boxes is not None and len(boxes) > 0:
                names = (
                    r.names
                    if hasattr(r, "names") and r.names is not None
                    else getattr(model, "names", {})
                )
                for box in boxes:
                    conf = math.ceil((box.conf[0] * 100)) / 100
                    cls = int(box.cls[0])
                    raw = names.get(cls, str(cls)) if isinstance(names, dict) else (
                        names[cls] if cls < len(names) else str(cls)
                    )
                    current_class = str(raw)
                    violation_classes = [
                        "NO_helmet",
                        "NO_Vest",
                        "NO_goggles",
                        "NO_SafetyShoes",
                        "NO_Gloves",
                    ]
                    if conf > 0.5 and current_class in violation_classes:
                        try:
                            face_dir = os.path.join(MEDIA_ROOT, "face_detect")
                            os.makedirs(face_dir, exist_ok=True)
                            cv2.imwrite(
                                os.path.join(face_dir, f"output{curr_datetime}.jpg"),
                                annotated_img,
                            )
                            cv2.imwrite(os.path.join(face_dir, "output.jpg"), annotated_img)
                        except Exception as write_error:
                            print(f"[file_analysis] save violation image: {write_error}")
                    break
            ok, buffer = cv2.imencode(
                ".jpg", annotated_img, [cv2.IMWRITE_JPEG_QUALITY, LIVE_MJPEG_JPEG_QUALITY]
            )
            if not ok or buffer is None:
                continue
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
            )
    except Exception as e:
        print(f"[file_analysis] stream error: {e}")
    finally:
        cap.release()
        video_processing_stop_requested = False
        video_processing_active = False
        current_processing_video_path = None
        current_processing_type = None


def generate_processed_frames3(video_path: str):
    """Zone PPE MJPEG: FILE_ANALYSIS_FRAME_SKIP, RESIZE, infer, JPEG; paced to file FPS (see demo2)."""
    global video_processing_stop_requested, model
    global video_processing_active, current_processing_video_path, current_processing_type
    _ensure_model_loaded()
    CONF_THRES = 0.25
    IOU_THRES = 0.45
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        video_processing_active = False
        current_processing_video_path = None
        current_processing_type = None
        return
    period = _file_analysis_frame_period_sec(cap)
    wall_clock: List[float] = []
    frame_counter = 0
    W, H = RESIZE[0], RESIZE[1]
    x_mid = W // 2
    divider = [x_mid, 0, x_mid, H - 1]
    zone_names = ("LEFT", "RIGHT")
    x1, y1, x2, y2 = divider
    CLR_LINE = (255, 255, 255)

    def point_side_of_line(px, py, ax1, ay1, ax2, ay2):
        return (ax2 - ax1) * (py - ay1) - (ay2 - ay1) * (px - ax1)

    def draw_label(img, text, x, y, color=(255, 255, 255), bg=(0, 0, 0)):
        (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (x, y - th - 6), (x + tw + 6, y + 2), bg, -1)
        cv2.putText(img, text, (x + 3, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    try:
        while True:
            if video_processing_stop_requested:
                break
            _file_analysis_wait_frame_slot(period, wall_clock)
            success, frame = cap.read()
            if not success:
                break
            frame_counter += 1
            if _file_analysis_should_skip_frame(frame_counter):
                continue
            frame = _file_analysis_resize_for_inference(frame)
            if frame is None or frame.size == 0:
                continue
            r0 = _file_analysis_infer(frame, conf=CONF_THRES, iou=IOU_THRES, classes=None)
            if r0 is None:
                continue
            annotated = frame.copy()
            dets = r0.boxes
            names = r0.names if hasattr(r0, "names") and r0.names is not None else model.names
            if dets is not None and len(dets) > 0:
                for i in range(len(dets)):
                    xyxy = dets.xyxy[i].cpu().tolist()
                    cls = int(dets.cls[i].cpu().item())
                    conf = float(dets.conf[i].cpu().item())
                    if isinstance(names, dict):
                        class_name = str(names.get(cls, str(cls)))
                    else:
                        class_name = str(names[cls]) if cls < len(names) else str(cls)
                    px1, py1, px2, py2 = [int(c) for c in xyxy]
                    if class_name.lower() == "person":
                        pcx, pcy = (px1 + px2) / 2, (py1 + py2) / 2
                        sign = point_side_of_line(pcx, pcy, x1, y1, x2, y2)
                        zone = zone_names[0] if sign > 0 else zone_names[1] if sign < 0 else "ON_LINE"
                        if zone == zone_names[1]:
                            cv2.rectangle(annotated, (px1, py1), (px2, py2), (0, 255, 0), 2)
                            draw_label(annotated, "OK", px1, py1 - 10, color=(255, 255, 255), bg=(0, 255, 0))
                        else:
                            cv2.rectangle(annotated, (px1, py1), (px2, py2), (0, 255, 0), 2)
                            cv2.putText(
                                annotated,
                                f"{class_name} {conf:.2f}",
                                (px1, py1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,
                                (0, 255, 0),
                                2,
                            )
                            draw_label(
                                annotated,
                                f"Zone: {zone}",
                                px1,
                                py1 - 30,
                                color=(255, 255, 255),
                                bg=(0, 0, 255),
                            )
                    else:
                        dcx, dcy = (px1 + px2) / 2, (py1 + py2) / 2
                        sign = point_side_of_line(dcx, dcy, x1, y1, x2, y2)
                        zone = zone_names[0] if sign > 0 else zone_names[1] if sign < 0 else "ON_LINE"
                        if zone == zone_names[0] or sign == 0:
                            cv2.rectangle(annotated, (px1, py1), (px2, py2), (0, 255, 0), 2)
                            cv2.putText(
                                annotated,
                                f"{class_name} {conf:.2f}",
                                (px1, py1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,
                                (0, 255, 0),
                                2,
                            )
                            if conf > 0.5 and class_name in [
                                "NO_helmet",
                                "NO_Vest",
                                "NO_goggles",
                                "NO_safetyshoes",
                            ]:
                                zdir = os.path.join(MEDIA_ROOT, "zone_based")
                                os.makedirs(zdir, exist_ok=True)
                                ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                                cv2.imwrite(os.path.join(zdir, f"output_{ts}.jpg"), annotated)
                                cv2.imwrite(os.path.join(zdir, "output.jpg"), annotated)
            cv2.line(annotated, (int(x1), int(y1)), (int(x2), int(y2)), CLR_LINE, 2)
            draw_label(
                annotated,
                "AUTO DIVIDER (VERTICAL)",
                int((x1 + x2) / 2),
                int((y1 + y2) / 2) - 6,
                color=(0, 0, 0),
                bg=(255, 255, 255),
            )
            cv2.putText(
                annotated,
                f"Zone Analysis: {zone_names[0]} / {zone_names[1]}",
                (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            _, buffer = cv2.imencode(
                ".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, LIVE_MJPEG_JPEG_QUALITY]
            )
            if buffer is None:
                continue
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
            )
    except Exception as e:
        print(f"[file_analysis] zone stream error: {e}")
    finally:
        cap.release()
        video_processing_stop_requested = False
        video_processing_active = False
        current_processing_video_path = None
        current_processing_type = None


def generate_processed_frames4(video_path: str):
    """Class-filtered YOLO MJPEG: FILE_ANALYSIS_FRAME_SKIP, RESIZE, infer, JPEG, FPS pacing (demo2/3)."""
    global video_processing_stop_requested, model
    global video_processing_active, current_processing_video_path, current_processing_type
    _ensure_model_loaded()
    selected_classes = list(file_analysis_selected_classes) or ["helmet", "shoes", "pvc_suit"]

    ALIASES = {
        "person": {"person", "Person"},
        "helmet": {"helmet", "hardhat", "safety_helmet", "Helmet"},
        "safety_vest": {"vest", "safety_vest", "Safety_Vestr"},
        "no_helmet": {"no_helmet", "no_safety_helmet", "no_hardhat", "NO_helmet"},
        "no_safety_vest": {"no_vest", "no_safety_vest", "NO_Vestr"},
        "pvc_suit": {"pvc_suit", "suit"},
        "no_pvc_suit": {"no_pvc_suit", "no_suit"},
        "shoes": {"shoes", "safety_shoes", "boots", "Safety Shoes"},
        "goggles": {"goggles", "safety_goggles", "glasses", "eye_protection", "Safety Goggles"},
        "no_safety_shoes": {"no_shoes", "NO_safetyshoes", "no_boots", "no_safety_shoes"},
        "no_goggles": {"no_goggles", "NO_goggles", "no_eye_protection", "no_safety_goggles"},
    }

    def canonicalize(name: str) -> str:
        n = name.lower().replace(" ", "_")
        for canon, synonyms in ALIASES.items():
            if n == canon or n in synonyms:
                return canon
        return n

    detect_classes_names: Set[str] = set()
    required_ppe: Set[str] = set()
    if selected_classes:
        user_ppe_types = set()
        for cls_name in selected_classes:
            cn = canonicalize(cls_name)
            if cn.startswith("no_"):
                user_ppe_types.add(cn[3:])
            else:
                user_ppe_types.add(cn)
        required_ppe = user_ppe_types
        for ppe_type in required_ppe:
            detect_classes_names.add(ppe_type)
            detect_classes_names.add(f"no_{ppe_type}")
    else:
        required_ppe = {"helmet", "shoes", "goggles", "safety_vest", "pvc_suit"}
        for ppe_type in required_ppe:
            detect_classes_names.add(ppe_type)
            detect_classes_names.add(f"no_{ppe_type}")
    detect_classes_names.add("person")

    names = model.names
    model_class_map: dict = {}
    if isinstance(names, dict):
        for idx, name in names.items():
            model_class_map[canonicalize(str(name))] = int(idx)
    else:
        for idx, name in enumerate(names):
            model_class_map[canonicalize(str(name))] = idx

    detect_class_indices: List[int] = []
    for name in detect_classes_names:
        if name in model_class_map:
            detect_class_indices.append(model_class_map[name])

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        video_processing_active = False
        current_processing_video_path = None
        current_processing_type = None
        return
    period = _file_analysis_frame_period_sec(cap)
    wall_clock: List[float] = []
    frame_count = 0
    try:
        while True:
            if video_processing_stop_requested:
                break
            _file_analysis_wait_frame_slot(period, wall_clock)
            try:
                success, frame = cap.read()
                if not success:
                    break
                frame_count += 1
                if _file_analysis_should_skip_frame(frame_count):
                    continue
                frame = _file_analysis_resize_for_inference(frame)
                if frame is None or frame.size == 0:
                    continue
                cls_arg = detect_class_indices if detect_class_indices else None
                r0 = _file_analysis_infer(
                    frame, conf=0.3, iou=0.5, classes=cls_arg
                )
                if r0 is None:
                    continue
                annotated_frame = _annotate_frame(frame, r0, filter_by_selected=False)
                _, buffer = cv2.imencode(
                    ".jpg",
                    annotated_frame,
                    [cv2.IMWRITE_JPEG_QUALITY, LIVE_MJPEG_JPEG_QUALITY],
                )
                if buffer is None:
                    continue
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
                )
            except Exception as frame_error:
                print(f"[file_analysis] class-based frame error: {frame_error}")
                continue
    except Exception as e:
        print(f"[file_analysis] class-based stream error: {e}")
    finally:
        cap.release()
        video_processing_stop_requested = False
        video_processing_active = False
        current_processing_video_path = None
        current_processing_type = None


# -----------------------------------------------------------------------------
# Class filter & frame annotation (shared by YOLO worker, display, HTTP/WebSocket)
# -----------------------------------------------------------------------------
def _get_effective_violation_names() -> Set[str]:
    """User selects PPE type (e.g. Helmet) = show VIOLATIONS (NO_helmet). Returns normalized model class names to show."""
    if not selected_class_names:
        return set()
    out: Set[str] = set()
    for sel in selected_class_names:
        sel_n = (sel or "").lower().replace(" ", "_")
        if sel_n in USER_SELECTION_TO_VIOLATION:
            for v in USER_SELECTION_TO_VIOLATION[sel_n]:
                out.add(v.lower().replace(" ", "_"))
        else:
            out.add(sel_n)
    return out


def _class_matches_selected(cls_name: str) -> bool:
    """True if normalized model class name is one of the violation classes to show (PPE selection → NO_*)."""
    if not selected_class_names or not cls_name:
        return True
    cls_n = (cls_name or "").lower().replace(" ", "_")
    effective = _get_effective_violation_names()
    if cls_n in effective:
        return True
    for v in effective:
        if v in cls_n or cls_n in v:
            return True
    return False


def _annotate_frame_for_exception(frame, result, exception_type: str):
    """
    Draw only violation-class boxes on the frame so the emailed/exception image
    clearly shows what was violated. Violation boxes are drawn in RED with
    a 'VIOLATION: <type>' label so the receiver can understand from the photo.
    Uses the same person + spatial + confidence rules as the live pipeline.
    """
    annotated = frame.copy() if frame is not None else frame
    if annotated is None or result is None:
        return annotated
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return annotated
    xyxy = boxes.xyxy.cpu().numpy()
    confs = boxes.conf.cpu().numpy()
    cls_np = boxes.cls.cpu().numpy()
    classes = cls_np.astype(int)
    names = result.names if hasattr(result, "names") else getattr(model, "names", {})
    want = _normalize_exception_name_for_db(exception_type)
    person_boxes = _person_boxes_from_frame(xyxy, cls_np, confs, names)
    for (x1, y1, x2, y2), conf, cls_id in zip(xyxy, confs, classes):
        cls_idx = int(cls_id)
        raw = names.get(cls_idx, str(cls_idx)) if isinstance(names, dict) else (names[cls_idx] if cls_idx < len(names) else str(cls_idx))
        norm_name = _normalize_exception_name_for_db(raw)
        if norm_name not in VIOLATION_CLASSES_FOR_LOG:
            continue
        if norm_name != want:
            continue
        if not violation_passes_person_rules(norm_name, float(conf), np.asarray([x1, y1, x2, y2], dtype=np.float32), person_boxes):
            continue
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        label = f"VIOLATION: {norm_name} ({conf:.2f})"
        color = (0, 0, 255)  # BGR red
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(annotated, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
        cv2.putText(annotated, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    return annotated


def _annotate_frame(frame, result, filter_by_selected=True):
    """
    Draw detection boxes on frame. Uses result from DISPLAY queue (full YOLO result with .boxes).
    When filter_by_selected=True, only draws classes in selected_class_names (from start_live_detection).
    Uses flexible matching so model names (e.g. vest, safety_shoes) match frontend ids (safety_vest, shoes).
    Person boxes below PERSON_MIN_CONF are not drawn (avoids labeling clutter as Person 0.26, etc.).
    PPE violation classes (no_helmet, no_vest, …) are drawn only when violation_passes_person_rules
    is true — same rules engine as the live JSON / exception pipeline.
    """
    annotated = frame.copy()
    if result is None:
        return annotated
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return annotated
    xyxy = boxes.xyxy.cpu().numpy()
    confs = boxes.conf.cpu().numpy()
    cls_np = boxes.cls.cpu().numpy()
    classes = cls_np.astype(int)
    names = result.names if hasattr(result, "names") else getattr(model, "names", {})
    person_boxes = _person_boxes_from_frame(xyxy, cls_np, confs, names)
    for (x1, y1, x2, y2), conf, cls_id in zip(xyxy, confs, classes):
        cls_idx = int(cls_id)
        raw = names.get(cls_idx, str(cls_idx)) if isinstance(names, dict) else (names[cls_idx] if cls_idx < len(names) else str(cls_idx))
        norm = _normalize_exception_name_for_db(raw)
        if norm == "person" and float(conf) < PERSON_MIN_CONF:
            continue
        if norm in VIOLATION_CLASSES_FOR_LOG:
            if not violation_passes_person_rules(
                norm,
                float(conf),
                np.asarray([x1, y1, x2, y2], dtype=np.float32),
                person_boxes,
            ):
                continue
        if filter_by_selected and selected_class_names:
            cls_name = (raw or "").strip()
            if not _class_matches_selected(cls_name):
                continue
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        label = f"{raw} {conf:.2f}"
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(annotated, (x1, y1 - th - 4), (x1 + tw, y1), (0, 255, 0), -1)
        cv2.putText(annotated, label, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return annotated


# -----------------------------------------------------------------------------
# Input readers (RTSP and video file)
# -----------------------------------------------------------------------------
def rtsp_reader(rtsp_url: str, cam_id: int, input_queue: queue.Queue):
    """RTSP reader: feeds only this camera's queue (per-camera, no cross-cam blocking)."""
    cap = None
    frame_count = 0
    reconnect_attempts = 0

    while True:
        if pipeline_stop_event.is_set():
            if cap is not None:
                cap.release()
            return
        try:
            if cap is None or not cap.isOpened():
                if reconnect_attempts >= MAX_RECONNECT_ATTEMPTS:
                    time.sleep(RECONNECT_DELAY * 2)
                    reconnect_attempts = 0
                cap = cv2.VideoCapture(rtsp_url)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                if not cap.isOpened():
                    reconnect_attempts += 1
                    time.sleep(RECONNECT_DELAY)
                    continue
                reconnect_attempts = 0

            ret, frame = cap.read()
            if not ret:
                reconnect_attempts += 1
                cap.release()
                cap = None
                time.sleep(RECONNECT_DELAY)
                continue
            reconnect_attempts = 0
            frame_count += 1
            if frame_count % FRAME_SKIP != 0:
                continue
            if frame.shape[:2] != RESIZE[::-1]:
                frame = cv2.resize(frame, RESIZE, interpolation=cv2.INTER_LINEAR)

            try:
                input_queue.put_nowait((cam_id, frame))
            except queue.Full:
                try:
                    input_queue.get_nowait()
                    input_queue.put_nowait((cam_id, frame))
                    with stats_lock:
                        performance_stats['frames_dropped'] += 1
                except queue.Empty:
                    pass
        except Exception as e:
            if cap:
                cap.release()
            cap = None
            time.sleep(RECONNECT_DELAY)


# -----------------------------------------------------------------------------
# YOLO batch worker (round-robin per-cam → GPU → per-cam output/display queues)
# -----------------------------------------------------------------------------
def yolo_batch_worker():
    """Pull from each camera queue in turn; batch inference on GPU; push to display."""
    global _debug_frames_with_detections, _debug_batches_processed
    batch = []
    meta = []
    frames = []
    last_batch_time = time.time()

    while True:
        if pipeline_stop_event.is_set():
            break
        try:
            # Round-robin from per-camera queues (fair; no one cam blocks another)
            if not pipeline_input_queues:
                time.sleep(0.05)
                continue
            got_any = False
            for q in pipeline_input_queues:
                try:
                    cam_id, frame = q.get_nowait()
                    batch.append(frame)
                    meta.append(cam_id)
                    frames.append(frame)
                    got_any = True
                    with stats_lock:
                        performance_stats['frames_processed'] += 1
                except queue.Empty:
                    pass
            if not got_any:
                time.sleep(0.01)
            # Single stream: use short timeout so we don't cap at ~2 FPS (batch rarely fills to BATCH_SIZE)
            n_queues = len(pipeline_input_queues)
            timeout = SINGLE_STREAM_BATCH_TIMEOUT if n_queues == 1 else BATCH_TIMEOUT
            should_process = len(batch) >= BATCH_SIZE or (
                batch and (time.time() - last_batch_time) >= timeout
            )
            if should_process:
                # Process batch
                start_time = time.time()
                
                try:
                    # TensorRT .engine models are compiled for batch size 1 only - run inference per frame
                    results = []
                    with torch.inference_mode():
                        for frame in batch:
                            res = model([frame], device=DEVICE, stream=False, verbose=False)
                            results.append(res[0])
                    
                    # Per-camera: push to this camera's output/display queues only
                    nq = len(pipeline_display_queues)
                    for cam_id, res, frame_copy in zip(meta, results, frames):
                        idx = 0 if nq == 1 else min(cam_id, nq - 1)
                        disp = {"camera_id": cam_id, "frame": frame_copy, "result": res}
                        boxes = res.boxes
                        person_boxes: List[np.ndarray] = []
                        # Realtime detection JSON: enqueue per-frame detection summary (all classes)
                        if boxes is not None and len(boxes) > 0:
                            names = res.names if hasattr(res, "names") else getattr(model, "names", {})
                            xyxy_np = boxes.xyxy.cpu().numpy()
                            cls_np = boxes.cls.cpu().numpy()
                            conf_np = boxes.conf.cpu().numpy()
                            person_boxes = _person_boxes_from_frame(xyxy_np, cls_np, conf_np, names)
                            detections_json = []
                            timestamp = datetime.utcnow().isoformat()
                            for (x1, y1, x2, y2), cls_id, score in zip(xyxy_np, cls_np, conf_np):
                                cls_idx = int(cls_id)
                                if isinstance(names, dict):
                                    raw_name = names.get(cls_idx, str(cls_idx))
                                else:
                                    raw_name = names[cls_idx] if 0 <= cls_idx < len(names) else str(cls_idx)
                                norm_name = _normalize_exception_name_for_db(raw_name)
                                is_violation = violation_passes_person_rules(
                                    norm_name,
                                    float(score),
                                    np.asarray([x1, y1, x2, y2], dtype=np.float32),
                                    person_boxes,
                                )
                                detections_json.append(
                                    {
                                        "class_id": cls_idx,
                                        "class_name": raw_name,
                                        "normalized_class_name": norm_name,
                                        "score": float(score),
                                        "bbox_xyxy": [float(x1), float(y1), float(x2), float(y2)],
                                        "is_violation": is_violation,
                                    }
                                )
                                # Append to plain-text log only for actionable violations (person + spatial + conf rules)
                                if is_violation:
                                    try:
                                        os.makedirs(MEDIA_ROOT, exist_ok=True)
                                        with open(DETECTION_LOG_FILE, "a", encoding="utf-8") as f:
                                            f.write(
                                                f"{timestamp}\t{int(cam_id)}\t{cls_idx}\t{raw_name}\t"
                                                f"{norm_name}\t{float(score):.4f}\t"
                                                f"{float(x1):.2f}\t{float(y1):.2f}\t{float(x2):.2f}\t{float(y2):.2f}\t"
                                                f"1\n"
                                            )
                                    except Exception as log_err:
                                        print(f"[camera_dashboard] detection log write failed: {log_err}")
                            event = {
                                "camera_id": int(cam_id),
                                "timestamp": timestamp,
                                "detections": detections_json,
                            }
                            try:
                                detection_json_queue.put_nowait(event)
                            except queue.Full:
                                try:
                                    detection_json_queue.get_nowait()
                                    detection_json_queue.put_nowait(event)
                                except queue.Empty:
                                    pass
                            # Store detection frame to disk (throttled per camera)
                            now = time.time()
                            with _detection_frame_time_lock:
                                last = _last_detection_frame_time.get(cam_id, 0)
                                if now - last >= DETECTION_FRAME_THROTTLE_SECONDS:
                                    _last_detection_frame_time[cam_id] = now
                                    try:
                                        detection_frames_queue.put_nowait((frame_copy.copy(), res, event, cam_id))
                                    except queue.Full:
                                        try:
                                            detection_frames_queue.get_nowait()
                                            detection_frames_queue.put_nowait((frame_copy.copy(), res, event, cam_id))
                                        except queue.Empty:
                                            pass
                        if boxes is not None and len(boxes) > 0 and idx < len(pipeline_output_queues):
                            out_q = pipeline_output_queues[idx]
                            names = res.names if hasattr(res, "names") else getattr(model, "names", {})
                            xyxy_np = boxes.xyxy.cpu().numpy()
                            cls_np = boxes.cls.cpu().numpy()
                            conf_np = boxes.conf.cpu().numpy()
                            if selected_class_names:
                                keep = [i for i in range(len(cls_np)) if _class_matches_selected(str(names.get(int(cls_np[i]), cls_np[i])) or "")]
                                if keep:
                                    result_data = {"camera_id": cam_id, "boxes": xyxy_np[keep].tolist(), "classes": cls_np[keep].tolist(), "scores": conf_np[keep].tolist()}
                                    try:
                                        out_q.put_nowait(result_data)
                                    except queue.Full:
                                        try:
                                            out_q.get_nowait()
                                            out_q.put_nowait(result_data)
                                        except queue.Empty:
                                            pass
                            else:
                                result_data = {"camera_id": cam_id, "boxes": xyxy_np.tolist(), "classes": cls_np.tolist(), "scores": conf_np.tolist()}
                                try:
                                    out_q.put_nowait(result_data)
                                except queue.Full:
                                    try:
                                        out_q.get_nowait()
                                        out_q.put_nowait(result_data)
                                    except queue.Empty:
                                        pass
                        if idx < len(pipeline_display_queues):
                            dq = pipeline_display_queues[idx]
                            try:
                                dq.put_nowait(disp)
                            except queue.Full:
                                try:
                                    dq.get_nowait()
                                    dq.put_nowait(disp)
                                except queue.Empty:
                                    pass
                        if idx < len(pipeline_imshow_queues):
                            iq = pipeline_imshow_queues[idx]
                            try:
                                iq.put_nowait(disp)
                            except queue.Full:
                                try:
                                    iq.get_nowait()
                                    iq.put_nowait(disp)
                                except queue.Empty:
                                    pass
                        # Exception log: enqueue violations for DB insert (throttled, non-blocking)
                        if boxes is not None and len(boxes) > 0:
                            names = res.names if hasattr(res, "names") else getattr(model, "names", {})
                            xyxy_np = boxes.xyxy.cpu().numpy()
                            cls_np = boxes.cls.cpu().numpy()
                            conf_np = boxes.conf.cpu().numpy()
                            seen_violations = set()
                            for i in range(len(cls_np)):
                                raw = _raw_class_name_from_names(names, int(cls_np[i]))
                                name = _normalize_exception_name_for_db(raw)
                                if name not in VIOLATION_CLASSES_FOR_LOG:
                                    continue
                                if not violation_passes_person_rules(
                                    name,
                                    float(conf_np[i]),
                                    xyxy_np[i],
                                    person_boxes,
                                ):
                                    continue
                                seen_violations.add(name)
                            if seen_violations:
                                now = time.time()
                                time_occurred = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                with _exception_log_time_lock:
                                    for v in seen_violations:
                                        key = (cam_id, v)
                                        last = _last_exception_log_time.get(key, 0)
                                        if now - last >= EXCEPTION_LOG_THROTTLE_SECONDS:
                                            _last_exception_log_time[key] = now
                                            try:
                                                # Draw violation boxes here (result is valid); worker only saves this image
                                                annotated_frame = _annotate_frame_for_exception(frame_copy.copy(), res, v)
                                                exception_log_queue.put_nowait((cam_id, annotated_frame, v, time_occurred))
                                            except queue.Full:
                                                _log_exception_pipeline(
                                                    f"exception_log: queue full ({EXCEPTION_LOG_QUEUE_SIZE}), "
                                                    f"dropping {v} cam={cam_id}"
                                                )
                    
                    # Update performance stats and debug counters
                    inference_time = time.time() - start_time
                    with stats_lock:
                        performance_stats['batches_processed'] += 1
                        if inference_time > 0:
                            performance_stats['fps'] = len(batch) / inference_time
                    with _debug_lock:
                        _debug_batches_processed += 1
                        for res in results:
                            if res.boxes is not None and len(res.boxes) > 0:
                                _debug_frames_with_detections += 1
                        if _debug_batches_processed % 50 == 0 and _debug_batches_processed > 0:
                            print(f"[camera_dashboard] DEBUG YOLO: batches={_debug_batches_processed}, frames_with_detections={_debug_frames_with_detections}")
                    
                    # Clear batch
                    batch.clear()
                    meta.clear()
                    frames.clear()
                    last_batch_time = time.time()
                    
                except Exception as e:
                    print(f"[camera_dashboard] YOLO inference error: {e}")
                    # Clear batch on error
                    batch.clear()
                    meta.clear()
                    frames.clear()
                    last_batch_time = time.time()
                    continue
                    
        except Exception as e:
            print(f"[camera_dashboard] Batch worker error: {e}")
            continue


# -----------------------------------------------------------------------------
# Detection-frames worker (save annotated frames + JSON under media/detection_frames/)
# -----------------------------------------------------------------------------
def _detection_frames_worker():
    """Background thread: save frames with detections to media/detection_frames/ (annotated image + JSON)."""
    try:
        os.makedirs(DETECTION_FRAMES_DIR, exist_ok=True)
        print(f"[camera_dashboard] Detection-frames worker started; saving to: {DETECTION_FRAMES_DIR}")
    except Exception as e:
        print(f"[camera_dashboard] detection_frames: ERROR creating dir {DETECTION_FRAMES_DIR}: {e}")
        return
    while True:
        if pipeline_stop_event.is_set():
            break
        try:
            item = detection_frames_queue.get(timeout=1.0)
        except queue.Empty:
            continue
        frame, result, event, cam_id = item
        if frame is None:
            continue
        try:
            annotated = _annotate_frame(frame, result, filter_by_selected=False) if result is not None else frame
            ts = event.get("timestamp", datetime.utcnow().isoformat())
            safe_ts = re.sub(r"[^\d\-T:]", "_", ts)[:26]
            filename_base = f"detection_{safe_ts}_cam{cam_id}"
            image_path = os.path.join(DETECTION_FRAMES_DIR, f"{filename_base}.jpg")
            ok = cv2.imwrite(image_path, annotated)
            if not ok:
                print(f"[camera_dashboard] detection_frames: cv2.imwrite returned False for {image_path}")
                continue
            json_path = os.path.join(DETECTION_FRAMES_DIR, f"{filename_base}.json")
            with open(json_path, "w") as f:
                json.dump(event, f, indent=2)
            print(f"[camera_dashboard] detection_frames: saved {filename_base}.jpg")
        except Exception as e:
            print(f"[camera_dashboard] detection_frames: failed to save: {e}")
            continue


# -----------------------------------------------------------------------------
# Exception-log worker (queue → disk → DB + email enqueue)
# -----------------------------------------------------------------------------
def _exception_log_worker():
    """Background thread: consume exception_log_queue, save annotated frame (with violation boxes) to disk, INSERT into exception_logs."""
    try:
        os.makedirs(EXCEPTION_LOGS_DIR, exist_ok=True)
        _log_exception_pipeline(f"exception_log: Exception-log worker started; saving to: {EXCEPTION_LOGS_DIR}")
    except Exception as e:
        _log_exception_pipeline(f"exception_log: ERROR creating dir {EXCEPTION_LOGS_DIR}: {e}")
        return
    while True:
        if pipeline_stop_event.is_set():
            break
        try:
            item = exception_log_queue.get(timeout=1.0)
        except queue.Empty:
            continue
        # (cam_id, annotated_frame, exception_type, time_occurred) - cam_id is pipeline stream index
        cam_id, frame, exception_type, time_occurred = item[0], item[1], item[2], item[3]
        if frame is None or not exception_type:
            continue
        db_camera_id = _pipeline_index_to_db_camera_id(cam_id)
        safe_type = re.sub(r"[^\w\-]", "_", exception_type)[:32]
        safe_ts = re.sub(r"[^\d\-]", "_", time_occurred)[:20]
        filename = f"exception_{safe_ts}_{db_camera_id}_{safe_type}.jpg"
        image_path = os.path.join(EXCEPTION_LOGS_DIR, filename)
        try:
            cv2.imwrite(image_path, frame)
            _log_exception_pipeline(f"exception_log: saved image {filename} (with violation boxes)")
        except Exception as e:
            _log_exception_pipeline(f"exception_log: failed to save image: {e}")
            continue
        # exception_logs.incident_image is LONGBLOB: store JPEG bytes, not the path string.
        try:
            with open(image_path, "rb") as _imgf:
                image_blob = _imgf.read()
        except OSError as e:
            _log_exception_pipeline(f"exception_log: failed to read image for DB: {e}")
            continue
        if not image_blob:
            _log_exception_pipeline("exception_log: empty image file, skip DB insert")
            continue
        db = SessionLocal()
        try:
            et_id = _resolve_exception_type_id(db, exception_type)
            if et_id is None:
                continue
            try:
                t_occ = datetime.strptime(time_occurred, "%Y-%m-%d %H:%M:%S")
            except ValueError:
                t_occ = datetime.now()
            db.execute(
                text(
                    """
                    INSERT INTO exception_logs
                    (time_occurred, exception_type_id, incident_image, updated_at, camera_id)
                    VALUES (:t, :eid, :i, NOW(), :cid)
                    """
                ),
                {"t": t_occ, "eid": et_id, "i": image_blob, "cid": db_camera_id},
            )
            db.commit()
            _log_exception_pipeline(
                f"exception_log: DB insert OK for {exception_type} (exception_type_id={et_id}) "
                f"camera_id={db_camera_id}"
            )
            # Enqueue email with camera name + zone from camera (direct query on open session).
            try:
                em_name, em_zone = _camera_name_zone_from_db(db, db_camera_id)
                enqueue_violation_email(
                    camera_id=db_camera_id,
                    exception_type=exception_type,
                    image_path=image_path,
                    time_occurred=time_occurred,
                    camera_name=em_name,
                    zone_name=em_zone,
                )
            except Exception as e:
                # Do not break logging if email enqueue fails.
                _log_exception_pipeline(f"exception_log: failed to enqueue email: {e}")
        except Exception as e:
            _log_exception_pipeline(f"exception_log: DB insert failed: {e}")
            db.rollback()
        finally:
            db.close()


# -----------------------------------------------------------------------------
# Display (imshow) — pipeline_imshow_queues only; does not affect live_detection_feed
# -----------------------------------------------------------------------------
def display_worker():
    """Read per-camera imshow queues, draw boxes via _annotate_frame, cv2.imshow per camera."""
    latest_frames = {}
    last_display_time = {}
    min_display_interval = 1.0 / 30
    while True:
        if pipeline_stop_event.is_set():
            cv2.destroyAllWindows()
            break
        try:
            if not pipeline_imshow_queues:
                time.sleep(0.05)
                continue
            for dq in pipeline_imshow_queues:
                try:
                    data = dq.get_nowait()
                except queue.Empty:
                    continue
                cam_id = data["camera_id"]
                frame = data["frame"]
                result = data["result"]
                annotated_frame = _annotate_frame(frame, result, filter_by_selected=True)
                latest_frames[cam_id] = annotated_frame
            current_time = time.time()
            for cam_id, annotated_frame in latest_frames.items():
                if cam_id in last_display_time and (current_time - last_display_time[cam_id]) < min_display_interval:
                    continue
                cv2.imshow(f"Camera {cam_id}", annotated_frame)
                last_display_time[cam_id] = current_time
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("[camera_dashboard] Display stopped (q)")
                break
            time.sleep(0.01)
        except Exception as e:
            print(f"Display error: {e}")
    cv2.destroyAllWindows()


# -----------------------------------------------------------------------------
# Performance monitor (terminal: pipeline FPS, GPU FPS, dropped frames)
# -----------------------------------------------------------------------------
def performance_monitor():
    """Print minimal stats to terminal every 5s."""
    while True:
        if pipeline_stop_event.is_set():
            break
        time.sleep(5)
        with stats_lock:
            stats = performance_stats.copy()
        elapsed = time.time() - stats['last_update']
        if elapsed > 0 and stats['frames_processed'] > 0:
            pipeline_fps = stats['frames_processed'] / elapsed
            print(f"[camera_dashboard] FPS: {pipeline_fps:.1f} | GPU: {stats['fps']:.1f} | Dropped: {stats['frames_dropped']}")
        performance_stats['frames_processed'] = 0
        performance_stats['frames_dropped'] = 0
        performance_stats['batches_processed'] = 0
        performance_stats['last_update'] = time.time()


def start_pipeline(
    rtsp_urls: Optional[List[str]] = None,
    class_filter: Optional[List[str]] = None,
) -> None:
    """
    Start live RTSP detection (use rtsp_urls if provided, else all from config).
    class_filter: if None or empty -> show all classes; else only these class names are shown.
    Queues are created first so /live_detection_status shows running=true immediately; model loads after.
    """
    global pipeline_input_queues, pipeline_output_queues, pipeline_display_queues, pipeline_imshow_queues, selected_class_names
    global _debug_frames_with_detections, _debug_batches_processed
    pipeline_stop_event.clear()
    _debug_frames_with_detections = 0
    _debug_batches_processed = 0
    selected_class_names = set(c.lower().replace(" ", "_") for c in class_filter) if class_filter else None
    urls_to_use = rtsp_urls if rtsp_urls is not None else RTSP_URLS
    if not urls_to_use:
        return
    n_streams = len(urls_to_use)
    # Create queues before model load so /live_detection_status shows running=true immediately
    pipeline_input_queues[:] = [queue.Queue(maxsize=QUEUE_SIZE) for _ in range(n_streams)]
    pipeline_output_queues[:] = [queue.Queue(maxsize=QUEUE_SIZE) for _ in range(n_streams)]
    pipeline_display_queues[:] = [queue.Queue(maxsize=QUEUE_SIZE) for _ in range(n_streams)]
    pipeline_imshow_queues[:] = [queue.Queue(maxsize=QUEUE_SIZE) for _ in range(n_streams)]

    os.makedirs(EXCEPTION_LOGS_DIR, exist_ok=True)
    os.makedirs(DETECTION_FRAMES_DIR, exist_ok=True)

    _ensure_model_loaded()

    for i, url in enumerate(urls_to_use):
        threading.Thread(
            target=rtsp_reader,
            args=(url, i, pipeline_input_queues[i]),
            daemon=True,
            name=f"RTSP-{i}",
        ).start()

    # Start workers: YOLO → display_queues; exception log → DB; detection frames → disk
    threading.Thread(target=yolo_batch_worker, daemon=True, name="YOLO-Worker").start()
    threading.Thread(target=_exception_log_worker, daemon=True, name="Exception-Log-Worker").start()
    threading.Thread(target=_detection_frames_worker, daemon=True, name="Detection-Frames-Worker").start()
    # threading.Thread(target=display_worker, daemon=True, name="Display-Worker").start()
    threading.Thread(target=performance_monitor, daemon=True, name="Performance-Monitor").start()
    print(f"[camera_dashboard] Pipeline started | streams: {n_streams} | detection images: {DETECTION_FRAMES_DIR} | violations: {EXCEPTION_LOGS_DIR}")


# -----------------------------------------------------------------------------
# FastAPI router (included from main.py)
# -----------------------------------------------------------------------------
router = APIRouter(
    prefix="/api/camera_dashboard",
    tags=["camera_dashboard"],
    dependencies=[Depends(require_permission("camera-dashboard.view"))],
)

# MJPEG file-analysis feeds must be reachable without Authorization (browser <img> cannot send Bearer).
public_feed_router = APIRouter(prefix="/api/camera_dashboard", tags=["camera_dashboard"])


@router.get("/cameras")
def get_cameras():
    """Get list of all available cameras from camera."""
    try:
        camera_config = get_camera_config()
        cameras = {
            cid: {
                "name": cfg.get("name", f"Camera {cid}"),
                "type": cfg.get("type", "unknown"),
                "url": cfg.get("url"),
                "description": cfg.get("description", ""),
            }
            for cid, cfg in camera_config.items()
        }
        return {"cameras": cameras, "total_cameras": len(cameras)}
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={"status": "error", "message": f"Failed to fetch cameras: {str(e)}"},
        )


class StartBody(BaseModel):
    camera_id: Optional[str] = None  # single camera (frontend sends this)
    camera_ids: Optional[List[str]] = None  # omit = all RTSP
    classes: Optional[List[str]] = None  # omit or empty = show all; else only these class names (e.g. helmet, shoes)


def _mark_streams_started(camera_ids: List[str]):
    """Insert running rows into camera_streams for current live session."""
    db = SessionLocal()
    try:
        for cam_id in camera_ids:
            if not str(cam_id).isdigit():
                continue
            stream_url = f"/api/camera_dashboard/live_detection_feed?camera_id={cam_id}"
            db.execute(
                text(
                    """
                    INSERT INTO camera_streams (camera_id, video_feed_url, status, started_at)
                    VALUES (:camera_id, :video_feed_url, 'running', NOW())
                    """
                ),
                {"camera_id": int(cam_id), "video_feed_url": stream_url},
            )
        db.commit()
    except Exception as e:
        db.rollback()
        print(f"[camera_dashboard] camera_streams start insert failed: {e}")
    finally:
        db.close()


def _mark_streams_stopped(camera_ids: List[str]):
    """Mark latest running rows as stopped for active camera_ids."""
    if not camera_ids:
        return
    db = SessionLocal()
    try:
        for cam_id in camera_ids:
            if not str(cam_id).isdigit():
                continue
            db.execute(
                text(
                    """
                    UPDATE camera_streams
                    SET status = 'stopped', stopped_at = NOW()
                    WHERE camera_id = :camera_id AND status = 'running'
                    """
                ),
                {"camera_id": int(cam_id)},
            )
        db.commit()
    except Exception as e:
        db.rollback()
        print(f"[camera_dashboard] camera_streams stop update failed: {e}")
    finally:
        db.close()


def _camera_index(camera_id: str) -> int:
    """Map camera_id to pipeline_display_queues index. Returns 0 if not found or invalid."""
    global current_camera_ids
    try:
        idx = current_camera_ids.index(str(camera_id))
        if 0 <= idx < len(pipeline_display_queues):
            return idx
    except (ValueError, AttributeError):
        pass
    return 0


@router.get("/live_detection_status")
def get_live_detection_status():
    """Return whether live detection is running and which cameras are active."""
    running = not pipeline_stop_event.is_set() and len(pipeline_display_queues) > 0
    return {
        "running": running,
        "camera_ids": list(current_camera_ids),
        "selected_classes": list(selected_class_names) if selected_class_names else None,
        "feed_url_base": "/api/camera_dashboard/live_detection_feed",
        "feed_ws_base": "/api/camera_dashboard/live_detection_feed_ws",
    }


@router.get("/debug_storage")
def debug_storage():
    """
    Debug endpoint: storage paths, whether dirs exist, pipeline state, queue sizes, and sample files.
    Use this to see exactly where the code is broken.
    Exception DB pipeline messages are also written to media/exception_pipeline.log; tail is included here.
    """
    def list_dir_safe(path: str, max_files: int = 20) -> List[str]:
        try:
            if not os.path.isdir(path):
                return []
            names = sorted(os.listdir(path))[:max_files]
            return names
        except Exception:
            return []

    def tail_file(path: str, max_lines: int = 40) -> List[str]:
        try:
            if not os.path.isfile(path):
                return []
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                lines = f.readlines()
            return [ln.rstrip("\n\r") for ln in lines[-max_lines:]]
        except Exception:
            return []

    running = not pipeline_stop_event.is_set() and len(pipeline_display_queues) > 0
    with _debug_lock:
        batches = _debug_batches_processed
        frames_with_det = _debug_frames_with_detections
    return {
        "media_root": MEDIA_ROOT,
        "exception_logs_dir": EXCEPTION_LOGS_DIR,
        "detection_frames_dir": DETECTION_FRAMES_DIR,
        "exception_logs_dir_exists": os.path.isdir(EXCEPTION_LOGS_DIR),
        "detection_frames_dir_exists": os.path.isdir(DETECTION_FRAMES_DIR),
        "pipeline_running": running,
        "pipeline_display_queues_count": len(pipeline_display_queues),
        "exception_log_queue_size": exception_log_queue.qsize(),
        "detection_frames_queue_size": detection_frames_queue.qsize(),
        "debug_batches_processed": batches,
        "debug_frames_with_detections": frames_with_det,
        "exception_logs_sample_files": list_dir_safe(EXCEPTION_LOGS_DIR),
        "detection_frames_sample_files": list_dir_safe(DETECTION_FRAMES_DIR),
        "exception_pipeline_log_path": EXCEPTION_PIPELINE_LOG_PATH,
        "exception_pipeline_log_exists": os.path.isfile(EXCEPTION_PIPELINE_LOG_PATH),
        "exception_pipeline_log_tail": tail_file(EXCEPTION_PIPELINE_LOG_PATH, 40),
    }


@router.get("/live_detections_json")
def get_live_detections_json(max_items: int = 100):
    """
    Return and clear up to max_items most recent detection events from the realtime JSON queue.
    Each event contains camera_id, timestamp, and list of detections with bbox and scores.
    """
    items = []
    taken = 0
    # Drain the queue up to max_items to provide true queue semantics
    while taken < max_items:
        try:
            event = detection_json_queue.get_nowait()
        except queue.Empty:
            break
        items.append(event)
        taken += 1
    return {"count": len(items), "events": items}


@router.get("/live_detection_feed")
async def live_detection_feed(camera_id: str = "0", quality: int = 82, draw_all_classes: bool = False):
    """
    Stream MJPEG from the selected camera's DISPLAY queue (frame + YOLO result).
    - camera_id: which camera's queue to read.
    - quality: JPEG quality 1-100 (default 82).
    - draw_all_classes: if False (default), draw only classes the user selected from frontend (start_live_detection); if True, draw all.
    """
    if pipeline_stop_event.is_set() or not pipeline_display_queues:
        raise HTTPException(status_code=503, detail="Live detection not running")
    idx = _camera_index(camera_id)
    display_queue = pipeline_display_queues[idx]  # has {"frame", "result"} with full detection boxes
    q = max(1, min(100, quality))

    def generate():
        while not pipeline_stop_event.is_set():
            data = None
            try:
                while True:
                    data = display_queue.get_nowait()
            except queue.Empty:
                pass
            if data is None:
                try:
                    data = display_queue.get(timeout=0.25)
                except Exception:
                    continue
            frame = data.get("frame")
            result = data.get("result")
            if frame is None:
                continue
            annotated = _annotate_frame(frame, result, filter_by_selected=(not draw_all_classes))
            _, jpeg = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, q])
            if jpeg is None:
                continue
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg.tobytes() + b"\r\n")

    return StreamingResponse(
        generate(),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={"Cache-Control": "no-store, no-cache, must-revalidate", "Pragma": "no-cache", "X-Content-Type-Options": "nosniff"},
    )


@router.websocket("/live_detection_feed_ws")
async def live_detection_feed_ws(websocket: WebSocket, camera_id: str = "0"):
    """
    Real-time feed over WebSocket. Two modes:
    - If client sends {"mode": "webrtc"}, perform WebRTC signaling (offer/answer) and stream via WebRTC (requires aiortc).
    - Otherwise stream JPEG frames as binary for low-latency canvas rendering.
    """
    # WebSocket auth: Authorization header only.
    auth = websocket.headers.get("authorization")
    raw_token = None
    if auth and auth.lower().startswith("bearer "):
        raw_token = auth.split(" ", 1)[1].strip()

    if not raw_token:
        await websocket.close(code=4401)
        return

    try:
        decoded = decode_access_token(raw_token)
        role = (decoded.get("role") or "").strip().lower()
        user_id = int(decoded.get("sub"))
    except Exception:
        await websocket.close(code=4401)
        return

    db = SessionLocal()
    try:
        allowed_keys = get_user_allowed_page_keys(db=db, user_id=user_id, role=role)
    finally:
        db.close()

    if "camera-dashboard.view" not in allowed_keys:
        await websocket.close(code=4403)
        return

    await websocket.accept()
    if pipeline_stop_event.is_set() or not pipeline_display_queues:
        await websocket.send_json({"error": "Live detection not running"})
        await websocket.close()
        return
    idx = _camera_index(camera_id)
    display_queue = pipeline_display_queues[idx]
    loop = asyncio.get_event_loop()

    try:
        # Optional first message: client may send {"mode": "webrtc"} for WebRTC; else we stream JPEG
        first = await asyncio.wait_for(websocket.receive_json(), timeout=2.0)
        use_webrtc = first.get("mode") == "webrtc"
    except (asyncio.TimeoutError, WebSocketDisconnect, Exception):
        use_webrtc = False

    if use_webrtc:
        try:
            from aiortc import RTCPeerConnection, RTCSessionDescription, MediaStreamTrack
            from av import VideoFrame

            class DisplayQueueTrack(MediaStreamTrack):
                kind = "video"

                def __init__(self, queue, ev_loop):
                    super().__init__()
                    self._queue = queue
                    self._loop = ev_loop
                    self._last_pts = 0

                async def recv(self):
                    while not pipeline_stop_event.is_set():
                        try:
                            data = await self._loop.run_in_executor(None, lambda: self._queue.get(timeout=0.3))
                        except Exception:
                            await asyncio.sleep(0.03)
                            continue
                        frame = data.get("frame")
                        result = data.get("result")
                        if frame is None:
                            continue
                        annotated = _annotate_frame(frame, result, filter_by_selected=True)
                        av_frame = VideoFrame.from_ndarray(annotated, format="bgr24")
                        av_frame.pts = self._last_pts
                        av_frame.time_base = "1/30"
                        self._last_pts += 1
                        return av_frame
                    raise Exception("Pipeline stopped")

            pc = RTCPeerConnection()
            track = DisplayQueueTrack(display_queue, loop)
            pc.addTrack(track)
            offer = await pc.createOffer()
            await pc.setLocalDescription(offer)
            await websocket.send_json({"type": "offer", "sdp": pc.localDescription.sdp})
            answer_msg = await websocket.receive_json()
            await pc.setRemoteDescription(RTCSessionDescription(sdp=answer_msg["sdp"], type=answer_msg["type"]))
            # Keep connection open so WebRTC can stream
            while not pipeline_stop_event.is_set() and pc.connectionState != "failed":
                await asyncio.sleep(0.5)
            await pc.close()
        except ImportError:
            await websocket.send_json({"error": "WebRTC (aiortc) not installed"})
        except Exception as e:
            await websocket.send_json({"error": str(e)})
        finally:
            try:
                await websocket.close()
            except Exception:
                pass
        return

    # Fallback: stream JPEG binary over WebSocket (optimized: drain to latest, fixed quality)
    _ws_quality = 82
    try:
        while not pipeline_stop_event.is_set():
            data = None
            def _drain_latest():
                d = None
                try:
                    while True:
                        d = display_queue.get_nowait()
                except queue.Empty:
                    pass
                return d
            data = await loop.run_in_executor(None, _drain_latest)
            if data is None:
                try:
                    data = await loop.run_in_executor(None, lambda: display_queue.get(timeout=0.2))
                except Exception:
                    continue
            frame = data.get("frame")
            result = data.get("result")
            if frame is None:
                continue
            annotated = _annotate_frame(frame, result, filter_by_selected=True)
            _, jpeg = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, _ws_quality])
            if jpeg is None:
                continue
            try:
                await websocket.send_bytes(jpeg.tobytes())
            except Exception:
                break
    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await websocket.send_json({"error": str(e)})
        except Exception:
            pass
    finally:
        try:
            await websocket.close()
        except Exception:
            pass


@router.post(
    "/start_live_detection",
    dependencies=[Depends(require_permission("camera-dashboard.view"))],
)
def api_start_live_detection(body: Optional[StartBody] = None):
    """Start: body.camera_id (single) or body.camera_ids for chosen RTSP (omit = all); body.classes to filter by class (omit = all)."""
    global current_camera_ids
    b = body or StartBody()
    # Create media folders and an empty detections log file immediately
    # so they exist even if the pipeline thread fails later or no detections occur.
    try:
        os.makedirs(EXCEPTION_LOGS_DIR, exist_ok=True)
        os.makedirs(DETECTION_FRAMES_DIR, exist_ok=True)
        _ensure_detection_log_file()
        print(
            f"[camera_dashboard] Media dirs ready: "
            f"exception_logs={EXCEPTION_LOGS_DIR}, "
            f"detection_frames={DETECTION_FRAMES_DIR}, "
            f"detection_log_file={DETECTION_LOG_FILE}"
        )
    except Exception as e:
        print(f"[camera_dashboard] ERROR creating media dirs or log file: {e}")
    camera_config = get_camera_config()
    # Support both camera_id (single, from frontend) and camera_ids (list); normalize to strings for config lookup
    raw_ids = b.camera_ids if (b.camera_ids and len(b.camera_ids) > 0) else ([b.camera_id] if b.camera_id else None)
    ids = [str(i) for i in raw_ids] if raw_ids else None
    rtsp_urls = get_rtsp_urls(ids)
    current_camera_ids = list(ids) if ids else sorted(
        [k for k, cfg in camera_config.items() if cfg.get("type") == "rtsp" and cfg.get("url")]
    )
    def _run_pipeline():
        try:
            start_pipeline(rtsp_urls, b.classes)
        except Exception as e:
            print(f"[camera_dashboard] Pipeline thread ERROR: {e}")
    threading.Thread(target=_run_pipeline, daemon=True).start()
    # Feed URL for selected camera (single-cam: first in list)
    feed_camera_id = current_camera_ids[0] if current_camera_ids else "0"
    feed_url = f"/api/camera_dashboard/live_detection_feed?camera_id={feed_camera_id}"
    feed_ws_url = f"/api/camera_dashboard/live_detection_feed_ws?camera_id={feed_camera_id}"
    _mark_streams_started(current_camera_ids)
    return {
        "status": "success",
        "feed_url": feed_url,
        "feed_ws_url": feed_ws_url,
        "camera_ids": current_camera_ids,
        "camera_name": camera_config.get(feed_camera_id, {}).get("name", f"Camera {feed_camera_id}"),
    }


@router.post(
    "/stop_live_detection",
    dependencies=[Depends(require_permission("camera-dashboard.view"))],
)
def api_stop_live_detection():
    """Signal the live detection pipeline to stop; workers will exit and display windows close."""
    pipeline_stop_event.set()
    _mark_streams_stopped(current_camera_ids)
    return {"status": "success"}


# --- File upload analysis (demo2, demo3, demo4) — matches frontend FileAnalysis.jsx + apiConfig ---


@router.post("/demo2")
async def file_analysis_demo2(file: UploadFile = File(...)):
    global video_processing_active, current_processing_type, current_processing_video_path
    global video_processing_stop_requested
    if not file.filename:
        raise HTTPException(status_code=400, detail={"status": "error", "error": "No selected file"})
    if not _allowed_upload_file(file.filename):
        raise HTTPException(status_code=400, detail={"status": "error", "error": "Invalid file type"})
    try:
        os.makedirs(UPLOAD_FOLDER, exist_ok=True)
        filename = _secure_upload_filename(file.filename)
        sample_path = os.path.abspath(os.path.join(UPLOAD_FOLDER, filename))
        content = await file.read()
        with open(sample_path, "wb") as f:
            f.write(content)
        video_processing_active = True
        current_processing_type = "general"
        current_processing_video_path = sample_path
        video_processing_stop_requested = False
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"output_{timestamp}_{filename}"
        vp = quote(sample_path, safe="")
        video_feed_url = f"/api/camera_dashboard/video_feed2?video_path={vp}"
        download_url = f"/static/uploads/{output_filename}"
        return {
            "status": "success",
            "video_feed_url": video_feed_url,
            "download_url": download_url,
            "message": "File uploaded successfully",
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={"status": "error", "error": f"Processing failed: {str(e)}"},
        )


@router.post("/demo3")
async def file_analysis_demo3(file: UploadFile = File(...)):
    global video_processing_active, current_processing_type, current_processing_video_path, video_processing_stop_requested
    if not file.filename:
        raise HTTPException(status_code=400, detail={"status": "error", "error": "No selected file"})
    if not _allowed_upload_file(file.filename):
        raise HTTPException(status_code=400, detail={"status": "error", "error": "Invalid file type"})
    try:
        os.makedirs(UPLOAD_FOLDER, exist_ok=True)
        filename = _secure_upload_filename(file.filename)
        sample_path = os.path.abspath(os.path.join(UPLOAD_FOLDER, filename))
        content = await file.read()
        with open(sample_path, "wb") as f:
            f.write(content)
        video_processing_active = True
        current_processing_type = "zone"
        current_processing_video_path = sample_path
        video_processing_stop_requested = False
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"output_{timestamp}_{filename}"
        vp = quote(sample_path, safe="")
        video_feed_url = f"/api/camera_dashboard/video_feed3?video_path={vp}"
        download_url = f"/static/uploads/{output_filename}"
        return {
            "status": "success",
            "video_feed_url": video_feed_url,
            "download_url": download_url,
            "message": "File uploaded successfully",
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={"status": "error", "error": f"Processing failed: {str(e)}"},
        )


@router.post("/demo4")
async def file_analysis_demo4(
    file: UploadFile = File(...),
    classes: Optional[str] = Form("[]"),
):
    global video_processing_active, current_processing_type, current_processing_video_path, video_processing_stop_requested
    global file_analysis_selected_classes
    if not file.filename:
        raise HTTPException(status_code=400, detail={"status": "error", "error": "No selected file"})
    if not _allowed_upload_file(file.filename):
        raise HTTPException(status_code=400, detail={"status": "error", "error": "Invalid file type"})
    try:
        selected_classes = json.loads(classes) if classes else []
        if not isinstance(selected_classes, list):
            selected_classes = ["helmet", "shoes", "pvc_suit"]
    except json.JSONDecodeError:
        selected_classes = ["helmet", "shoes", "pvc_suit"]
    try:
        os.makedirs(UPLOAD_FOLDER, exist_ok=True)
        filename = _secure_upload_filename(file.filename)
        sample_path = os.path.abspath(os.path.join(UPLOAD_FOLDER, filename))
        content = await file.read()
        with open(sample_path, "wb") as f:
            f.write(content)
        video_processing_active = True
        current_processing_type = "class"
        current_processing_video_path = sample_path
        video_processing_stop_requested = False
        file_analysis_selected_classes = [str(x) for x in selected_classes]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"output_{timestamp}_{filename}"
        vp = quote(sample_path, safe="")
        video_feed_url = f"/api/camera_dashboard/video_feed4?video_path={vp}"
        download_url = f"/static/uploads/{output_filename}"
        return {
            "status": "success",
            "video_feed_url": video_feed_url,
            "download_url": download_url,
            "message": f"File uploaded successfully with classes: {file_analysis_selected_classes}",
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={"status": "error", "error": f"Processing failed: {str(e)}"},
        )


def _mjpeg_stream(gen):
    return StreamingResponse(
        gen,
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0",
        },
    )


@public_feed_router.get("/video_feed2")
def file_analysis_video_feed2(video_path: str):
    safe = _safe_path_under_upload(unquote(video_path))
    if not safe:
        raise HTTPException(status_code=404, detail={"status": "error", "error": "Invalid video path"})
    return _mjpeg_stream(generate_processed_frames2(safe))


@public_feed_router.get("/video_feed3")
def file_analysis_video_feed3(video_path: str):
    safe = _safe_path_under_upload(unquote(video_path))
    if not safe:
        raise HTTPException(status_code=404, detail={"status": "error", "error": "Invalid video path"})
    return _mjpeg_stream(generate_processed_frames3(safe))


@public_feed_router.get("/video_feed4")
def file_analysis_video_feed4(video_path: str):
    safe = _safe_path_under_upload(unquote(video_path))
    if not safe:
        raise HTTPException(status_code=404, detail={"status": "error", "error": "Invalid video path"})
    return _mjpeg_stream(generate_processed_frames4(safe))


@router.post("/stop_video_processing")
def file_analysis_stop_video_processing():
    global video_processing_active, current_processing_type, current_processing_video_path, video_processing_stop_requested
    try:
        video_processing_stop_requested = True
        video_processing_active = False
        current_processing_type = None
        current_processing_video_path = None
        return {"status": "success", "message": "Video processing stopped successfully"}
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "status": "error",
                "message": "Failed to stop video processing",
                "error": str(e),
            },
        )


@router.get("/video_processing_status")
def file_analysis_video_processing_status():
    return {
        "processing_active": video_processing_active,
        "processing_type": current_processing_type,
        "video_path": current_processing_video_path,
        "stop_requested": video_processing_stop_requested,
    }

