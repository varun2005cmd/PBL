# =============================================================================
# app/ml/__init__.py
# Face Recognition Authentication ML Package
# =============================================================================

from app.ml.face_detector import detect_face
from app.ml.liveness import check_liveness
from app.ml.embedder import generate_embedding
from app.ml.recognizer import recognize_user, train_classifier
from app.ml.pipeline import authenticate_user

__all__ = [
    "detect_face",
    "check_liveness",
    "generate_embedding",
    "recognize_user",
    "train_classifier",
    "authenticate_user",
]
