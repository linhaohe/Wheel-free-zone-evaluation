import cv2, os

cap = cv2.VideoCapture("./calibration/test.mp4")
os.makedirs("./calibration/frame", exist_ok=True)
count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    cv2.imwrite(f"./calibration/frame/frame_{count:04d}.png", frame)
    count += 1

cap.release()