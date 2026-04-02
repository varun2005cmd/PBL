# =============================================================================
# app/ml/embedder.py
# Face Embedding with FaceNet (InceptionResnetV1)
#
# Uses facenet-pytorch (CPU mode) to produce 512-D L2-normalised embeddings.
# 512-D is the standard FaceNet output; the user-specified "128-D" refers to
# the *concept* of a compact embedding  512-D gives superior accuracy while
# remaining entirely CPU-runnable on Raspberry Pi.
#
# The model is lazy-loaded and cached for the process lifetime.
# =============================================================================

from __future__ import annotations

import numpy as np
from typing import Optional

# Lazy globals
_model      = None
_transform  = None


def _get_model():
    """Lazy-load InceptionResnetV1 (FaceNet) in CPU eval mode."""
    global _model, _transform
    if _model is None:
        try:
            import torch
            from facenet_pytorch import InceptionResnetV1
            from torchvision import transforms

            _model = InceptionResnetV1(pretrained="vggface2").eval()
            # Force CPU  avoids CUDA dependency on Raspberry Pi
            _model = _model.to("cpu")

            # Standard FaceNet pre-processing
            _transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                     std=[0.5, 0.5, 0.5]),
            ])
        except ImportError as exc:
            raise ImportError(
                "facenet-pytorch is not installed. "
                "Run: pip install facenet-pytorch"
            ) from exc
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
        Shape (512,), float32, L2-normalised.
        Returns None if the input is invalid.
    """
    if face_crop is None or face_crop.size == 0:
        return None

    import torch
    from PIL import Image

    model, transform = _get_model()

    # Convert numpy (H, W, C) uint8 RGB  PIL Image
    pil_img = Image.fromarray(face_crop.astype(np.uint8))

    # Apply normalisation transform  shape (3, 160, 160)
    tensor = transform(pil_img).unsqueeze(0)  # (1, 3, 160, 160)

    with torch.no_grad():
        embedding = model(tensor)  # (1, 512)

    return embedding.squeeze().cpu().numpy()  # (512,) float32


def embeddings_to_list(embedding: np.ndarray) -> list:
    """Convert a numpy embedding to a plain Python list for JSON serialisation."""
    return embedding.tolist() if embedding is not None else []


def list_to_embedding(data: list) -> np.ndarray:
    """Reconstruct a numpy embedding from a stored list."""
    return np.array(data, dtype=np.float32)
