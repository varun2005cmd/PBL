# =============================================================================
# app/ml/embedder.py  –  v2 (hardened)
#
# Fixes applied:
#   • _model_lock prevents race conditions during parallel requests
#   • generate_embedding() wraps entire inference in try/except
#   • face_crop dtype / shape validated before torch conversion
#   • torch.inference_mode() instead of no_grad() (faster + safer on Pi)
#   • All exceptions logged (no silent failures)
# =============================================================================

from __future__ import annotations

import logging
import os
import threading
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# Lazy globals – guarded by _model_lock
_model      = None
_transform  = None
_model_lock = threading.Lock()


def _get_model():
    """Lazy-load InceptionResnetV1 (FaceNet) in CPU eval mode.  Thread-safe."""
    global _model, _transform
    if _model is not None:
        return _model, _transform

    with _model_lock:
        # Double-check inside lock
        if _model is not None:
            return _model, _transform

        try:
            cache_dir = Path(__file__).parent / "model_store" / "torch_cache"
            os.environ.setdefault("TORCH_HOME", str(cache_dir))

            import torch
            from facenet_pytorch import InceptionResnetV1
            from torchvision import transforms

            pretrained = os.environ.get("FACENET_PRETRAINED", "vggface2")
            logger.info("Loading FaceNet (pretrained=%s) …", pretrained)
            _model = InceptionResnetV1(pretrained=pretrained).eval().to("cpu")

            _transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ])
            logger.info("FaceNet loaded successfully.")
        except ImportError as exc:
            raise ImportError(
                "facenet-pytorch is not installed. Run: pip install facenet-pytorch"
            ) from exc
        except Exception as exc:
            logger.error("FaceNet model load failed: %s", exc)
            raise

    return _model, _transform


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_embedding(face_crop: np.ndarray) -> Optional[np.ndarray]:
    """
    Compute a 512-D L2-normalised face embedding from a pre-cropped face.

    Parameters
    ----------
    face_crop : np.ndarray
        RGB image with shape (160, 160, 3) and dtype uint8, as produced by
        ``detect_face()``.

    Returns
    -------
    np.ndarray | None
        Shape (512,), float32, L2-normalised.  None on any failure.
    """
    if face_crop is None:
        logger.debug("generate_embedding: received None.")
        return None

    if not isinstance(face_crop, np.ndarray):
        logger.debug("generate_embedding: input is not ndarray (type=%s).", type(face_crop).__name__)
        return None

    if face_crop.size == 0 or face_crop.ndim != 3 or face_crop.shape[2] != 3:
        logger.debug("generate_embedding: invalid face_crop shape %s.", face_crop.shape)
        return None

    # Ensure uint8 (PIL requires it)
    if face_crop.dtype != np.uint8:
        face_crop = np.clip(face_crop, 0, 255).astype(np.uint8)

    # Normalize face crop lighting using CLAHE (L channel in LAB color space)
    try:
        import cv2
        lab = cv2.cvtColor(face_crop, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        lab = cv2.merge((l, a, b))
        face_crop = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    except Exception as exc:
        logger.debug("generate_embedding: CLAHE normalization skipped: %s", exc)

    try:
        import torch
        from PIL import Image

        model, transform = _get_model()

        # Test Time Augmentation (TTA) for max accuracy: Original + Flipped
        pil_img_orig = Image.fromarray(face_crop)
        pil_img_flip = Image.fromarray(cv2.flip(face_crop, 1))

        tensor_orig = transform(pil_img_orig).unsqueeze(0)   # (1, 3, 160, 160)
        tensor_flip = transform(pil_img_flip).unsqueeze(0)   # (1, 3, 160, 160)

        batch_tensor = torch.cat([tensor_orig, tensor_flip], dim=0) # (2, 3, 160, 160)

        with torch.inference_mode():
            embeddings = model(batch_tensor)               # (2, 512)

        # Average the two embeddings and re-normalize
        embeddings_np = embeddings.cpu().numpy()
        avg_embedding = np.mean(embeddings_np, axis=0)     # (512,)
        result = avg_embedding / np.linalg.norm(avg_embedding) # L2 normalize

        # Sanity-check output shape
        if result.shape != (512,):
            logger.error("generate_embedding: unexpected output shape %s.", result.shape)
            return None

        return result

    except Exception as exc:
        logger.error("generate_embedding: inference failed: %s", exc)
        return None


def embeddings_to_list(embedding: np.ndarray) -> list:
    """Convert a numpy embedding to a plain Python list for JSON serialisation."""
    if embedding is None:
        return []
    try:
        return embedding.tolist()
    except Exception:
        return []


def list_to_embedding(data: list) -> np.ndarray:
    """Reconstruct a numpy embedding from a stored list."""
    return np.array(data, dtype=np.float32)


def warmup() -> None:
    """Force-load the FaceNet model into memory at startup."""
    try:
        _get_model()
        logger.info("FaceNet (InceptionResnetV1) warmed up.")
    except Exception as exc:
        logger.warning("FaceNet warmup failed (non-fatal): %s", exc)
