import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from app.app import create_app

app = create_app()
print("create_app(): OK")

from app.ml.face_detector import warmup

warmup()
print("face_detector warmup: OK")
