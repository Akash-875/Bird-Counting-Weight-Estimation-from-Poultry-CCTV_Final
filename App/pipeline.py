
import cv2
import time
import json
from math import hypot
from collections import deque
from ultralytics import YOLO
from app.weight_estimator import estimate_weight_index

# ---------------- CONFIG ----------------
MODEL_PATH = "/content/chicken_training/yolov8_chicken_cpu/weights/best.pt"
IMG_SIZE = 1536
CONF_THRESH = 0.20
MOVE_THRESHOLD = 8
SMOOTH_WINDOW = 5
SLICE_OVERLAP = 0.35
DENSITY_GRID = 60
# --------------------------------------


# -------- IMAGE ENHANCEMENT (CLAHE) --------
def enhance_frame(frame):
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    return cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2BGR)


# -------- 9-SLICE FRAME TILING --------
def slice_frame(frame):
    h, w, _ = frame.shape
    dx = int(w * SLICE_OVERLAP / 3)
    dy = int(h * SLICE_OVERLAP / 3)

    slices = []
    for i in range(3):
        for j in range(3):
            x1 = max(0, j * w // 3 - dx)
            y1 = max(0, i * h // 3 - dy)
            x2 = min(w, (j + 1) * w // 3 + dx)
            y2 = min(h, (i + 1) * h // 3 + dy)
            slices.append((x1, y1, x2, y2))
    return slices


# -------- TOTAL BIRD ESTIMATION (DENSITY SNAPSHOT) --------
def estimate_total_birds_density(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    count = 0

    for y in range(0, h, DENSITY_GRID):
        for x in range(0, w, DENSITY_GRID):
            cell = gray[y:y + DENSITY_GRID, x:x + DENSITY_GRID]
            if cell.size == 0:
                continue
            if cell.var() > 120:
                count += 1
    return count


# -------- MAIN PIPELINE --------
def process_video(video_path, output_video_path, output_json_path):

    model = YOLO(MODEL_PATH)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out = cv2.VideoWriter(
        output_video_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        int(fps),
        (W, H)
    )

    prev_centers = {}
    active_ids = set()
    count_history = deque(maxlen=SMOOTH_WINDOW)
    timeline = []

    frame_no = 0
    start_time = time.time()
    total_birds_estimate = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = enhance_frame(frame)
        detections = []

        # ---------- SLICED YOLO DETECTION ----------
        for sx1, sy1, sx2, sy2 in slice_frame(frame):
            crop = frame[sy1:sy2, sx1:sx2]

            results = model.track(
                crop,
                persist=True,
                tracker="bytetrack.yaml",
                imgsz=IMG_SIZE,
                conf=CONF_THRESH,
                iou=0.5,
                verbose=False
            )

            if results[0].boxes is None:
                continue

            boxes = results[0].boxes.xyxy.cpu().numpy()
            ids = results[0].boxes.id
            ids = ids.cpu().numpy() if ids is not None else []

            for box, tid in zip(boxes, ids):
                x1, y1, x2, y2 = map(int, box)
                detections.append(
                    (x1 + sx1, y1 + sy1, x2 + sx1, y2 + sy1, tid)
                )

        active_count = 0
        weight_sum = 0

        # ---------- MOTION FILTER ----------
        for x1, y1, x2, y2, tid in detections:
            if tid is None:
                continue

            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            if tid not in prev_centers:
                prev_centers[tid] = (cx, cy)
                continue

            px, py = prev_centers[tid]
            prev_centers[tid] = (cx, cy)

            if hypot(cx - px, cy - py) < MOVE_THRESHOLD:
                continue

            active_ids.add(int(tid))
            active_count += 1

            w, h = x2 - x1, y2 - y1
            wt = estimate_weight_index(w, h)
            weight_sum += wt

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame,
                f"ID:{int(tid)} W:{wt}",
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 255, 0),
                1
            )

        # ---------- SMOOTHING ----------
        count_history.append(active_count)
        smooth_active = int(sum(count_history) / len(count_history))
        avg_weight = round(weight_sum / smooth_active, 2) if smooth_active else 0

        # ---------- TOTAL BIRDS SNAPSHOT ----------
        if frame_no % int(fps * 5) == 0:
            total_birds_estimate = estimate_total_birds_density(frame)

        timeline.append({
            "timestamp_sec": round(frame_no / fps, 2),
            "active_birds": smooth_active,
            "average_weight_index": avg_weight
        })

        cv2.putText(
            frame,
            f"Active Birds: {smooth_active}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2
        )

        out.write(frame)
        frame_no += 1

    cap.release()
    out.release()

    response = {
        "estimated_total_birds": total_birds_estimate,
        "total_active_birds": len(active_ids),
        "processing_fps": round(frame_no / (time.time() - start_time), 2),
        "counts": timeline,
        "weight_estimates": "relative_weight_index",
        "artifacts": {
            "annotated_video": output_video_path
        }
    }

    with open(output_json_path, "w") as f:
        json.dump(response, f, indent=2)

    return response
