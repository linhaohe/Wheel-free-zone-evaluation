import csv
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO

# --- Folders ---
VIDEO_DIR = Path("video/Goessmann")   # every video in here is processed
CSV_DIR = Path("csv/Goessmann")       # one CSV per video is written here (created if missing)
MODEL_PATH = Path("model/yolo26s.pt") 
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".m4v"}
WINDOW = "ROI YOLO"

# --- Batch settings ---
REUSE_FIRST_ROI = True  # True = draw the ROI once and reuse it for every video
SKIP_EXISTING = False    # True = skip videos that already have a CSV

# --- Counting settings ---
TIMESPAN_S = 30    # time period (video seconds) for the "unique IDs in last N s" count

#Camera parameters for pixel to ground conversion
CAM_HEIGHT = 3.55   # meters
VFOV_DEG = 64.0   # vertical field of view in degrees
HFOV_DEG = 124.0   # horizontal field of view in degrees
pitch_deg = 30.0  # camera pitch angle in degrees (positive = looking down)


# --- Speed settings ---
IMGSZ = 960        # inference size; lower (e.g. 480 or 320) = faster, less accurate
FRAME_SKIP = 0     # frames to skip between processed ones (1 = every 2nd frame)
USE_GPU = torch.cuda.is_available()
DEVICE = 0 if USE_GPU else "cpu"

CSV_FIELDS = [
    "video_time", "Density",
]


def draw_roi(frame, title=""):
    """Let the user click polygon points on `frame`.

    Controls:
      Left click  - add a point
      Right click - remove the last point
      r           - reset all points
      Enter/Space - confirm (needs at least 3 points)
      q           - skip this video
      Esc         - stop the whole batch
    Returns a list of (x, y) tuples, or None if this video is skipped.
    """
    pts = []

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            pts.append((x, y))
        elif event == cv2.EVENT_RBUTTONDOWN and pts:
            pts.pop()

    cv2.namedWindow(WINDOW)
    cv2.setMouseCallback(WINDOW, on_mouse)

    while True:
        canvas = frame.copy()

        if len(pts) >= 2:
            cv2.polylines(canvas, [np.array(pts, np.int32)], len(pts) >= 3,
                          (0, 255, 255), 2)
        for p in pts:
            cv2.circle(canvas, p, 4, (0, 0, 255), -1)

        cv2.putText(
            canvas,
            "L-click: add | R-click: undo | r: reset | Enter: confirm | q: skip | Esc: stop all",
            (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2,
        )
        if title:
            cv2.putText(canvas, title, (10, 55), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 255, 0), 2)
        cv2.imshow(WINDOW, canvas)

        key = cv2.waitKey(20) & 0xFF
        if key in (13, 32) and len(pts) >= 3:  # Enter / Space
            return pts
        elif key == ord("r"):
            pts.clear()
        elif key == ord("q"):
            return None
        elif key == 27:  # Esc
            raise KeyboardInterrupt

def pixel_to_ground(u, v, W, H, hfov_deg, cam_height, pitch_deg, yaw_deg=0.0, f=None):
    cx, cy = W / 2, H / 2
    if f is None:
        f = cx / np.radians(hfov_deg / 2)      # equidistant model: r = f * theta

    dx, dy = u - cx, v - cy
    r = np.hypot(dx, dy)
    if r < 1e-9:
        d_cam = np.array([0.0, 0.0, 1.0])
    else:
        theta = r / f                           # angle from optical axis (rad)
        s = np.sin(theta) / r
        d_cam = np.array([dx * s, dy * s, np.cos(theta)])

    th = np.radians(pitch_deg)
    R_pitch = np.array([
        [1,  0,           0          ],
        [0, -np.cos(th), -np.sin(th)],
        [0, -np.sin(th),  np.cos(th)],
    ])
    ps = np.radians(yaw_deg)
    R_yaw = np.array([
        [ np.cos(ps), 0, np.sin(ps)],
        [0, 1, 0],
        [-np.sin(ps), 0, np.cos(ps)],
    ])
    d = R_yaw @ (R_pitch @ d_cam)

    if d[1] >= 0:
        return None
    t = -cam_height / d[1]
    return np.array([0, cam_height, 0]) + t * d

def track_video(model, cap, roi, rows):
    """Run tracking on an opened video; append one dict per video-second to `rows`.

    q = stop this video (rows so far are kept), Esc = stop the whole batch.
    """
    ids_seen_in_roi = set()  # every unique PERSON tracker ID that has entered the ROI
    bicycle_ids = set()      # every unique BICYCLE tracker ID that has entered the ROI

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    world = [pixel_to_ground(x, y, frame_width, frame_height, HFOV_DEG, CAM_HEIGHT, pitch_deg) for x, y in roi]
    ground_xz = np.array([[p[0], p[2]] for p in world], dtype=np.float32)
    area = cv2.contourArea(ground_xz)
    # print(" ROI world Area (m^2):", cv2.contourArea(ground_xz))

    frames_read = 0          # counts skipped frames too, so video time stays correct
    last_seen = {}           # person tracker ID -> video time (s), coordinates world, and speed last in the ROI
    next_record_time = TIMESPAN_S

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        frames_read += 1

        # Cheaply discard frames to speed up processing
        for _ in range(FRAME_SKIP):
            cap.grab()
            frames_read += 1

        video_time = frames_read / fps  # seconds of video played so far

        results = model.track(
            frame, persist=True, classes=[0, 1], tracker="bytetrack.yaml",
            conf=0.65, iou=0.4, imgsz=IMGSZ, device=DEVICE, quantize=32,
            verbose=False,
        )

        cv2.polylines(frame, [roi], True, (0, 255, 255), 2)

        current_in_roi = set()       # people in the ROI right now
        bicycles_in_roi_now = set()  # bicycles in the ROI right now

        for r in results:
            for box in r.boxes:
                if box.id is None:
                    continue
                track_id = int(box.id.item())

                x1, y1, x2, y2 = box.xyxy[0].tolist()
                reference_point = ((x1 + x2) / 2, y2)  # bottom-center of the box

                # 1 = inside, 0 = on edge, -1 = outside
                inside = cv2.pointPolygonTest(roi, reference_point, False)

                if inside >= 0:
                    class_name = model.names[int(box.cls.item())]
                    confidence = float(box.conf.item())
                    speed = 0.0
                    label = f"{class_name} {confidence:.0%}"

                    if class_name == "person" or class_name == "bicycle":
                        if class_name == "person":
                            current_in_roi.add(track_id)
                        else:
                            bicycles_in_roi_now.add(track_id)
                            bicycle_ids.add(track_id)
                        ids_seen_in_roi.add(track_id)

                        world_coordinates = pixel_to_ground(
                            reference_point[0], reference_point[1],
                            frame_width, frame_height,
                            HFOV_DEG, CAM_HEIGHT, pitch_deg,
                        )

                        if track_id in last_seen:
                            prev_time, prev_world, prev_speed = last_seen[track_id]
                            dt = video_time - prev_time
                            if world_coordinates is not None and prev_world is not None and 0 < dt <= 1.0:
                                inst = float(np.linalg.norm(world_coordinates - prev_world)) / dt
                                speed = 0.7 * prev_speed + 0.3 * inst   # smoothed, m/s
                            else:
                                speed = prev_speed

                        last_seen[track_id] = (video_time, world_coordinates, speed)
                        label = f"{class_name} {confidence:.0%} {speed:.1f} m/s"
                        if class_name == "person":
                            color = (0, 255, 0)       # green = person
                        else:
                            color = (255, 255, 255)
                    else:
                        color = (255, 255, 255)

                    cv2.rectangle(
                        frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2,
                    )
                    cv2.putText(
                        frame, label, (int(x1), max(int(y1) - 8, 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2,
                    )

        # Forget people not seen within the time period
        cutoff = video_time - 5
        for tid in [t for t, ts in last_seen.items() if ts[0] < cutoff ]:
            del last_seen[tid]

        # Record one row per time interval 
        # if video_time >= next_record_time:
        #     rows.append({
        #         "video_time": round(video_time/60, 2),
        #         "Density": len(last_seen)/area,  # people per m^2
        #     })
        #     next_record_time += TIMESPAN_S
        speeds = [
            spd for tid, (_, _, spd) in last_seen.items()
            if tid in current_in_roi and spd is not None
        ]
        average_speed = float(np.mean(speeds)) if speeds else 0.0
        max_speed = float(np.max(speeds)) if speeds else 0.0
        cv2.putText(
            frame,
            f"In ROI now: {len(current_in_roi)} | Unique total: {len(ids_seen_in_roi)}",
            (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2,
        )
        cv2.putText(
            frame, f"Density in person/m²: {len(last_seen)/area:.2f}",
            (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2,
        )
        cv2.putText(
            frame,
            f"Average Speed: {average_speed:.1f} m/s",
            (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2,
        )
        cv2.putText(
            frame,
            f"Max Speed: {max_speed:.1f} m/s",
            (10, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2,
        )
        cv2.imshow(WINDOW, frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break                    # stop this video, go to the next
        elif key == 27:              # Esc
            raise KeyboardInterrupt  # stop everything
        elif key == ord(" "):
            cv2.waitKey(0)           # pause until any key is pressed


def write_csv(rows, csv_path):
    if not rows:
        print(f"  No data recorded, nothing written for {csv_path.name}")
        return
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Saved {len(rows)} rows to {csv_path}")


def main():
    if not VIDEO_DIR.is_dir():
        print(f"Video folder not found: {VIDEO_DIR.resolve()}")
        return

    videos = sorted(
        p for p in VIDEO_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in VIDEO_EXTS
    )
    if not videos:
        print(f"No videos found in {VIDEO_DIR.resolve()}")
        return

    CSV_DIR.mkdir(exist_ok=True)
    model = YOLO(MODEL_PATH)  # loaded once, reused for every video
    shared_roi = None

    try:
        for i, video_path in enumerate(videos, 1):
            csv_path = CSV_DIR / f"{video_path.stem}.csv"
            print(f"[{i}/{len(videos)}] {video_path.name}")

            if SKIP_EXISTING and csv_path.exists():
                print("  CSV already exists, skipping")
                continue

            cap = cv2.VideoCapture(str(video_path))
            success, first_frame = cap.read()
            if not success:
                print("  Could not read video, skipping")
                cap.release()
                continue

            # Step 1: ROI (drawn per video, or reused from the first one)
            if REUSE_FIRST_ROI and shared_roi is not None:
                roi = shared_roi
            else:
                pts = draw_roi(first_frame, title=video_path.name)
                if pts is None:
                    print("  Skipped")
                    cap.release()
                    continue
                roi = np.array(pts, np.int32)
                shared_roi = roi
                print("  ROI points:", pts)
       

            # Rewind and start a fresh tracker so IDs don't carry over between videos
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            model.predictor = None

            # Step 2: track, and always save what we have (even on Esc / crash)
            rows = []
            try:
                track_video(model, cap, roi, rows)
            finally:
                write_csv(rows, csv_path)
                cap.release()
    except KeyboardInterrupt:
        print("Stopped by user.")
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()