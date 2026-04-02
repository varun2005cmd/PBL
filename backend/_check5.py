import sys
sys.path.insert(0, r'c:\Users\Palash\PBL\backend')
from app.app import create_app
app = create_app()
print('create_app: OK')
from app.ml.face_detector import warmup
warmup()
print('face_detector: OK')
from app.ml.embedder import _get_model
_get_model()
print('FaceNet: OK')
from app.ml.recognizer import train_classifier
print('recognizer import: OK')
print('ALL PIPELINE COMPONENTS: READY')
