import cv2
import numpy as np
import torch
from ultralytics import YOLO

VIDEO_PATH = "video.mp4"
WINDOW = "ROI YOLO"

# --- Speed settings ---
IMGSZ = 640        # inference size; lower (e.g. 480 or 320) = faster, less accurate
FRAME_SKIP = 0     # frames to skip between processed ones (1 = every 2nd frame)
USE_GPU = torch.cuda.is_available()
DEVICE = 0 if USE_GPU else "cpu"


def draw_roi(frame):
    """Let the user click polygon points on `frame`.

    Controls:
      Left click  - add a point
      Right click - remove the last point
      r           - reset all points
      Enter/Space - confirm (needs at least 3 points)
      q / Esc     - cancel
    Returns a list of (x, y) tuples, or None if cancelled.
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
            "L-click: add | R-click: undo | r: reset | Enter: confirm | q: quit",
            (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2,
        )
        cv2.imshow(WINDOW, canvas)

        key = cv2.waitKey(20) & 0xFF
        if key in (13, 32) and len(pts) >= 3:  # Enter / Space
            return pts
        elif key == ord("r"):
            pts.clear()
        elif key in (ord("q"), 27):  # q / Esc
            return None


def main():
    model = YOLO("yolo26s.pt")  # loaded once, before the loop
    cap = cv2.VideoCapture(VIDEO_PATH)

    success, first_frame = cap.read()
    if not success:
        print(f"Could not read video: {VIDEO_PATH}")
        return

    # Step 1: draw the ROI on the first frame
    pts = draw_roi(first_frame)
    if pts is None:
        cap.release()
        cv2.destroyAllWindows()
        return

    roi = np.array(pts, np.int32)
    print("ROI points:", pts)

    # Rewind so processing starts from the first frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # Step 2: run tracking, keeping only people inside the ROI
    ids_seen_in_roi = set()  # every unique tracker ID that has entered the ROI
    id_map = {}              # tracker ID -> sequential display ID (1, 2, 3, ...)
    next_display_id = 1
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # Cheaply discard frames to speed up playback/processing
        for _ in range(FRAME_SKIP):
            cap.grab()

        # persist=True keeps track IDs consistent across frames; classes=[0] = person
        results = model.track(
            frame, persist=True, classes=[0], tracker="bytetrack.yaml",
            conf=0.5, iou=0.5, imgsz=IMGSZ, device=DEVICE, quantize=32,
            verbose=False,
        )

        # Draw the ROI outline on every frame
        cv2.polylines(frame, [roi], True, (0, 255, 255), 2)

        current_in_roi = set()

        for r in results:
            for box in r.boxes:
                # box.id is None when the tracker hasn't assigned an ID yet
                if box.id is None:
                    continue
                track_id = int(box.id.item())

                x1, y1, x2, y2 = box.xyxy[0].tolist()
                # Bottom-center of the box (good for "feet on the ground" checks)
                reference_point = ((x1 + x2) / 2, y2)

                # 1 = inside, 0 = on edge, -1 = outside
                inside = cv2.pointPolygonTest(roi, reference_point, False)

                if inside >= 0:
                    current_in_roi.add(track_id)
                    ids_seen_in_roi.add(track_id)

                    # Give each new person the next number: 1, 2, 3, ...
                    if track_id not in id_map:
                        id_map[track_id] = next_display_id
                        next_display_id += 1
                    display_id = id_map[track_id]

                    cv2.rectangle(
                        frame, (int(x1), int(y1)), (int(x2), int(y2)),
                        (0, 255, 0), 2,
                    )
                    cv2.putText(
                        frame, f"ID {display_id}", (int(x1), max(int(y1) - 8, 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2,
                    )

        cv2.putText(
            frame,
            f"In ROI now: {len(current_in_roi)} | Unique total: {len(ids_seen_in_roi)}",
            (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2,
        )

        cv2.imshow(WINDOW, frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()