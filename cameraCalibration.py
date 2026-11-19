import cv2
import numpy as np
from pathlib import Path

dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
board = cv2.aruco.CharucoBoard((11, 7), 0.031, 0.023, dictionary)
# board.setLegacyPattern(True)  # try this if nothing is detected
detector = cv2.aruco.CharucoDetector(board)

files = sorted(Path("./calibration/frame").glob("*.png"))
print(f"Found {len(files)} images")
if not files:
    raise SystemExit("No images found - check the path and extension")

obj_pts, img_pts, image_size = [], [], None
for f in files:
    img = cv2.imread(str(f))
    if img is None:
        continue
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    image_size = (int(gray.shape[1]), int(gray.shape[0]))  # (width, height)
    corners, ids, _, _ = detector.detectBoard(gray)
    if ids is not None and len(ids) >= 15:
        o, i = board.matchImagePoints(corners, ids)
        obj_pts.append(o.reshape(1, -1, 3).astype(np.float64))
        img_pts.append(i.reshape(1, -1, 2).astype(np.float64))

print(f"Using {len(obj_pts)} / {len(files)} frames")

K, D = np.zeros((3, 3)), np.zeros((4, 1))
flags = ( cv2.CALIB_RECOMPUTE_EXTRINSIC
    | cv2.CALIB_CHECK_COND
    | cv2.CALIB_FIX_SKEW)  # add CALIB_CHECK_COND back once stable

ret, K, D, rvecs, tvecs = cv2.fisheye.calibrate(
    obj_pts, img_pts, image_size, K, D, flags=flags,
    criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6))

print(f"RMS reprojection error: {ret:.4f} px")
print("K:\n", K)
print("D:\n", D.ravel())
# print("rvecs:\n", rvecs)
# print("tvecs:\n", tvecs)
np.savez("fisheye_calib.npz", K=K, D=D, image_size=image_size)