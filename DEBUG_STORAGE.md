# Debugging: Detection images not stored

Use this to find exactly where the pipeline or storage is broken.

---

## Step 1: Create folders as soon as you click Start

**What to do:** Start the backend, then call **Start detection** from the frontend (or `POST /api/camera_dashboard/start_live_detection` with `{"camera_ids": ["1"]}`).

**Check:** Open this URL in the browser (or use curl):

```
GET http://localhost:8000/api/camera_dashboard/debug_storage
```

**Interpret:**

| Field | Meaning |
|-------|--------|
| `exception_logs_dir_exists` | `true` = folder was created |
| `detection_frames_dir_exists` | `true` = folder was created |
| `exception_logs_dir` / `detection_frames_dir` | Full paths where images are saved |

- If both are **false**: backend failed to create the dirs (e.g. permission or path error). Check the **backend console** for `[camera_dashboard] ERROR creating media dirs: ...`.
- If both are **true**: folders exist. Go to Step 2.

---

## Step 2: Pipeline running

In the same `debug_storage` response:

| Field | Meaning |
|-------|--------|
| `pipeline_running` | `true` = detection pipeline is active |
| `pipeline_display_queues_count` | Number of camera queues (e.g. 1) |

- If **pipeline_running is false**: pipeline did not start or already stopped. Possible causes:
  - Model load failed (see backend console for `YOLO inference error` or `Pipeline thread ERROR`).
  - No RTSP URLs (wrong camera IDs). Check `GET /api/camera_dashboard/live_detection_status`: `camera_ids` should match what you sent.
- If **pipeline_running is true**: pipeline is running. Go to Step 3.

---

## Step 3: Frames with detections

In the same `debug_storage` response:

| Field | Meaning |
|-------|--------|
| `debug_batches_processed` | How many inference batches ran |
| `debug_frames_with_detections` | How many frames had at least one detection |

- If **debug_batches_processed is 0**: no frames are reaching YOLO. Possible causes:
  - RTSP not connected (camera offline or wrong URL). Check backend console for reconnection messages.
  - Input queues empty (readers not pushing frames).
- If **batches_processed > 0** but **frames_with_detections is 0**: YOLO runs but the model never detects anything (no person/PPE in view or model not trained for the scene). Detection **images** are only saved when there is at least one detection; try pointing the camera at a person with/without PPE.
- If **frames_with_detections > 0**: detections are happening. Go to Step 4.

---

## Step 4: Detection frames queue and files

In `debug_storage`:

| Field | Meaning |
|-------|--------|
| `detection_frames_queue_size` | Items waiting to be written to disk |

- If **frames_with_detections > 0** but **detection_frames_queue_size is always 0** and no files appear: throttle (one frame per camera every 5 seconds) may be delaying; wait at least 5 seconds and check again. Or the worker might have crashed: check backend console for `Detection-frames worker started` and `detection_frames: saved ...` or `detection_frames: failed to save: ...`.
- **detection_frames_sample_files**: list of files in `detection_frames_dir`. If it stays empty while frames_with_detections > 0, the detection_frames_worker is not writing (check console for errors).

**Backend console (what to look for):**

- On pipeline start: `Pipeline started | streams: 1 | detection images: ...`
- When detection-frames worker starts: `Detection-frames worker started; saving to: ...`
- When a detection image is saved: `detection_frames: saved detection_..._cam0.jpg`
- Every 50 batches: `DEBUG YOLO: batches=50, frames_with_detections=...`

---

## Step 5: Violation images (exception_logs)

Violation images (no_helmet, no_vest, etc.) are stored only when the model detects a **violation** and the 180-second throttle allows.

- **exception_logs_sample_files**: list of files in `exception_logs_dir`. Empty is normal if no violations were detected or throttle has not passed.
- Backend console: `exception_log: saved image exception_...` and `exception_log: DB insert OK for no_helmet` when a violation is saved.

---

## Quick checklist

1. Call `GET /api/camera_dashboard/debug_storage` after clicking Start.
2. Confirm `exception_logs_dir_exists` and `detection_frames_dir_exists` are **true** (if not, check console for media dir errors).
3. Confirm `pipeline_running` is **true** (if not, check model/RTSP and console).
4. Confirm `debug_batches_processed` increases over time (if 0, no frames reaching YOLO).
5. Confirm `debug_frames_with_detections` increases when the scene has people/PPE (if always 0, no detections → no detection images).
6. Check `detection_frames_sample_files` and the backend console for `detection_frames: saved` to confirm files are being written.

---

## Paths on your machine

After calling `debug_storage`, use the `exception_logs_dir` and `detection_frames_dir` values and open those folders in File Explorer. Example:

- `d:\Visual_Merge_new\visual_analytics_fastapi_backend\media\exception_logs`
- `d:\Visual_Merge_new\visual_analytics_fastapi_backend\media\detection_frames`
