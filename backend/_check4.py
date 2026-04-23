import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import cv2
import numpy as np

from app.ml.face_detector import detect_face, warmup

warmup()
print("warmup: OK")
dummy = np.zeros((480, 640, 3), dtype=np.uint8)
result = detect_face(dummy)
print("detect_face on blank frame (expect None):", result)
print("Pipeline import: PASS")
