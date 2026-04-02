import sys
sys.path.insert(0, r'c:\Users\Palash\PBL\backend')
from app.app import create_app
app = create_app()
print('create_app(): OK')
from app.ml.face_detector import warmup
warmup()
print('face_detector warmup: OK')
