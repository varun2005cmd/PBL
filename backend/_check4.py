import sys
sys.path.insert(0, r'c:\Users\Palash\PBL\backend')
from app.ml.face_detector import warmup, detect_face
warmup()
print('warmup: OK')
import numpy as np, cv2
dummy = np.zeros((480, 640, 3), dtype=np.uint8)
result = detect_face(dummy)
print('detect_face on blank frame (expect None):', result)
print('Pipeline import: PASS')
