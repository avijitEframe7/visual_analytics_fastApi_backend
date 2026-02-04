import os
import sys
import cv2
import time
import threading
import queue
import numpy as np
from collections import deque
from typing import Optional, List, Set
from ultralytics import YOLO
import torch
from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import asyncio
import json

from app.routes.camera_config import CAMERA_CONFIG, get_rtsp_urls

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



# ======================
# CONFIG
# ======================
RTSP_URLS = get_rtsp_urls()

# FPS / throughput: skip every Nth frame; resize before inference
FRAME_SKIP = 1
RESIZE = (640, 460)
BATCH_SIZE = 4
QUEUE_SIZE = 50
BATCH_TIMEOUT = 0.5          # (s) process partial batch after this (multi-camera)
SINGLE_STREAM_BATCH_TIMEOUT = 0.05  # (s) much shorter so single-camera doesn't wait 0.5s → ~2 FPS
MAX_RECONNECT_ATTEMPTS = 5
RECONNECT_DELAY = 2

# ======================
# QUEUES & LOCKS (all per-camera for max optimization)
# ======================
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

# Stop signal for live detection pipeline
pipeline_stop_event = threading.Event()

# Ordered camera ids for current pipeline (index = display queue index)
current_camera_ids: List[str] = []

# ======================
# YOLO MODEL (GPU; loaded on first /start_live_detection, not on import)
# ======================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "ML_models", "latest_16_1_2026.engine"))
model = None
DEVICE = "cuda:0"


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
    model = YOLO(MODEL_PATH)
    if MODEL_PATH.lower().endswith(".pt"):
        model.to(DEVICE)
        model.model.half()
    if MODEL_PATH.lower().endswith(".engine"):
        try:
            model(np.zeros((640, 640, 3), dtype=np.uint8), device=DEVICE, verbose=False)
        except Exception as e:
            if "nvinfer" in str(e) or "Could not find" in str(e):
                print("[camera_dashboard] TensorRT DLL not found. Add TensorRT bin to PATH or use .pt model.")
                sys.exit(1)
            raise
    print(f"[camera_dashboard] GPU: {gpu_name} | Model: {os.path.basename(MODEL_PATH)}")

# ======================
# OPTIMIZED RTSP READER
# ======================
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


# ======================
# VIDEO FILE READER
# ======================
def video_reader(video_path: str, cam_id: int, input_queue: queue.Queue):
    """Video file reader: feeds single queue until end of file."""
    if not os.path.isfile(video_path):
        return
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return
    frame_count = 0
    while True:
        if pipeline_stop_event.is_set():
            cap.release()
            return
        ret, frame = cap.read()
        if not ret:
            break
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
                    performance_stats["frames_dropped"] += 1
            except queue.Empty:
                pass
    cap.release()


# ======================
# YOLO WORKER (round-robin per-cam input → GPU → per-cam output/display queues)
# ======================
def yolo_batch_worker():
    """Pull from each camera queue in turn; batch inference on GPU; push to display."""
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
                        if len(boxes) > 0 and idx < len(pipeline_output_queues):
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
                    
                    # Update performance stats
                    inference_time = time.time() - start_time
                    with stats_lock:
                        performance_stats['batches_processed'] += 1
                        if inference_time > 0:
                            performance_stats['fps'] = len(batch) / inference_time
                    
                    # Clear batch
                    batch.clear()
                    meta.clear()
                    frames.clear()
                    last_batch_time = time.time()
                    
                except Exception as e:
                    print(f"YOLO inference error: {e}")
                    # Clear batch on error
                    batch.clear()
                    meta.clear()
                    frames.clear()
                    last_batch_time = time.time()
                    continue
                    
        except Exception as e:
            print(f"Batch worker error: {e}")
            continue


# ======================
# DISPLAY (imshow) – uses pipeline_imshow_queues only; does not affect live_detection_feed API
# ======================
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


# ======================
# PERFORMANCE (terminal: pipeline FPS, GPU inference FPS, dropped)
# ======================
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


VIDEO_CAM_ID = len(RTSP_URLS)  # virtual camera id used for video files


def start_pipeline(
    video_path: Optional[str] = None,
    rtsp_urls: Optional[List[str]] = None,
    class_filter: Optional[List[str]] = None,
) -> Optional[threading.Thread]:
    """
    Start the detection pipeline.
    - If video_path is provided and valid -> run video detection.
    - Else -> run live RTSP detection (use rtsp_urls if provided, else all from config).
    - class_filter: if None or empty -> show all classes; else only these class names are shown.
    Returns the video thread if video detection is started, otherwise None.
    """
    global pipeline_input_queues, pipeline_output_queues, pipeline_display_queues, pipeline_imshow_queues, selected_class_names
    pipeline_stop_event.clear()
    selected_class_names = set(c.lower().replace(" ", "_") for c in class_filter) if class_filter else None
    _ensure_model_loaded()
    urls_to_use = rtsp_urls if rtsp_urls is not None else RTSP_URLS
    video_thread: Optional[threading.Thread] = None
    n_streams = 1 if video_path else len(urls_to_use)
    pipeline_input_queues[:] = [queue.Queue(maxsize=QUEUE_SIZE) for _ in range(n_streams)]
    pipeline_output_queues[:] = [queue.Queue(maxsize=QUEUE_SIZE) for _ in range(n_streams)]
    pipeline_display_queues[:] = [queue.Queue(maxsize=QUEUE_SIZE) for _ in range(n_streams)]
    pipeline_imshow_queues[:] = [queue.Queue(maxsize=QUEUE_SIZE) for _ in range(n_streams)]

    if video_path:
        if not os.path.isfile(video_path):
            return None
        video_thread = threading.Thread(
            target=video_reader,
            args=(video_path, VIDEO_CAM_ID, pipeline_input_queues[0]),
            daemon=True,
            name="Video-Reader",
        )
        video_thread.start()
    else:
        if not urls_to_use:
            return None
        for i, url in enumerate(urls_to_use):
            threading.Thread(
                target=rtsp_reader,
                args=(url, i, pipeline_input_queues[i]),
                daemon=True,
                name=f"RTSP-{i}",
            ).start()

    # Start workers: YOLO → fills display_queues (API); imshow disabled (uncomment next line for local testing only)
    threading.Thread(target=yolo_batch_worker, daemon=True, name="YOLO-Worker").start()
    # threading.Thread(target=display_worker, daemon=True, name="Display-Worker").start()
    threading.Thread(target=performance_monitor, daemon=True, name="Performance-Monitor").start()
    print(f"[camera_dashboard] Pipeline started | streams: {n_streams} | API: GET /api/camera_dashboard/live_detection_feed?camera_id=N")
    return video_thread


# ======================
# FASTAPI ROUTER (entry from main.py)
# ======================
router = APIRouter(prefix="/api/camera_dashboard", tags=["camera_dashboard"])


@router.get("/cameras")
def get_cameras():
    """Get list of all available cameras from CAMERA_CONFIG."""
    try:
        cameras = {
            cid: {
                "name": cfg.get("name", f"Camera {cid}"),
                "type": cfg.get("type", "unknown"),
                "url": cfg.get("url"),
                "description": cfg.get("description", ""),
            }
            for cid, cfg in CAMERA_CONFIG.items()
        }
        return {"cameras": cameras, "total_cameras": len(cameras)}
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={"status": "error", "message": f"Failed to fetch cameras: {str(e)}"},
        )


class StartBody(BaseModel):
    video_path: Optional[str] = None
    camera_id: Optional[str] = None  # single camera (frontend sends this)
    camera_ids: Optional[List[str]] = None  # omit = all RTSP
    classes: Optional[List[str]] = None  # omit or empty = show all; else only these class names (e.g. helmet, shoes)


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


def _get_effective_violation_names() -> Set[str]:
    """User selects PPE type (e.g. Helmet) = show VIOLATIONS (NO_helmet). Returns set of normalized model class names to show."""
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
    """True if normalized model class name is one of the violation classes we want to show (user selected PPE = show NO_*)."""
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


def _annotate_frame(frame, result, filter_by_selected=True):
    """
    Draw detection boxes on frame. Uses result from DISPLAY queue (full YOLO result with .boxes).
    When filter_by_selected=True, only draws classes in selected_class_names (from start_live_detection).
    Uses flexible matching so model names (e.g. vest, safety_shoes) match frontend ids (safety_vest, shoes).
    """
    annotated = frame.copy()
    if result is None:
        return annotated
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return annotated
    xyxy = boxes.xyxy.cpu().numpy()
    confs = boxes.conf.cpu().numpy()
    classes = boxes.cls.cpu().numpy().astype(int)
    names = result.names if hasattr(result, "names") else getattr(model, "names", {})
    for (x1, y1, x2, y2), conf, cls_id in zip(xyxy, confs, classes):
        if filter_by_selected and selected_class_names:
            raw = names.get(cls_id, str(cls_id)) if isinstance(names, dict) else (names[cls_id] if cls_id < len(names) else str(cls_id))
            cls_name = (raw or "").strip()
            if not _class_matches_selected(cls_name):
                continue
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        label = f"{names.get(cls_id, str(cls_id))} {conf:.2f}" if isinstance(names, dict) else f"{cls_id} {conf:.2f}"
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(annotated, (x1, y1 - th - 4), (x1 + tw, y1), (0, 255, 0), -1)
        cv2.putText(annotated, label, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return annotated


@router.websocket("/live_detection_feed_ws")
async def live_detection_feed_ws(websocket: WebSocket, camera_id: str = "0"):
    """
    Real-time feed over WebSocket. Two modes:
    - If client sends {"mode": "webrtc"}, perform WebRTC signaling (offer/answer) and stream via WebRTC (requires aiortc).
    - Otherwise stream JPEG frames as binary for low-latency canvas rendering.
    """
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


@router.post("/start_live_detection")
def api_start_live_detection(body: Optional[StartBody] = None):
    """Start: body.video_path for file; body.camera_id (single) or body.camera_ids for chosen RTSP (omit = all); body.classes to filter by class (omit = all)."""
    global current_camera_ids
    b = body or StartBody()
    video_path = b.video_path
    # Support both camera_id (single, from frontend) and camera_ids (list)
    ids = b.camera_ids if (b.camera_ids and len(b.camera_ids) > 0) else ([b.camera_id] if b.camera_id else None)
    rtsp_urls = None if video_path else get_rtsp_urls(ids)
    if video_path:
        current_camera_ids = ["0"]
    else:
        current_camera_ids = list(ids) if ids else sorted(
            [k for k in CAMERA_CONFIG if CAMERA_CONFIG[k].get("type") == "rtsp" and CAMERA_CONFIG[k].get("url")]
        )
    threading.Thread(target=lambda: start_pipeline(video_path, rtsp_urls, b.classes), daemon=True).start()
    # Feed URL for selected camera (single-cam: first in list)
    feed_camera_id = current_camera_ids[0] if current_camera_ids else "0"
    feed_url = f"/api/camera_dashboard/live_detection_feed?camera_id={feed_camera_id}"
    feed_ws_url = f"/api/camera_dashboard/live_detection_feed_ws?camera_id={feed_camera_id}"
    return {
        "status": "success",
        "feed_url": feed_url,
        "feed_ws_url": feed_ws_url,
        "camera_ids": current_camera_ids,
        "camera_name": CAMERA_CONFIG.get(feed_camera_id, {}).get("name", f"Camera {feed_camera_id}"),
    }


@router.post("/stop_live_detection")
def api_stop_live_detection():
    """Signal the live detection pipeline to stop; workers will exit and display windows close."""
    pipeline_stop_event.set()
    return {"status": "success"}

