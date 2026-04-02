# =============================================================================
# app/ml/recognizer.py
# Face Recognition  SVM Classifier + Euclidean Distance Validation
#
# Two-stage recognition:
#   Stage 1  SVM classifier predicts the most likely identity.
#   Stage 2  Euclidean distance to the closest stored prototype embedding
#             must be below DISTANCE_THRESHOLD; otherwise the user is
#             reported as "unknown" even if SVM made a prediction.
#
# The trained SVM is persisted to disk so it survives server restarts.
# Call train_classifier() whenever users are enrolled/amended.
# =============================================================================

from __future__ import annotations

import os
import json
import pickle
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Where the trained SVM pickle is stored
_MODEL_DIR  = Path(__file__).parent / "model_store"
_SVM_PATH   = _MODEL_DIR / "svm_classifier.pkl"
_LABELS_PATH = _MODEL_DIR / "label_map.json"

# Euclidean distance: below this = known user, above = unknown
DISTANCE_THRESHOLD = 0.9   # tunable; lower = stricter

# SVM minimum probability to be considered a valid identity
SVM_PROB_THRESHOLD = 0.55

# Cached classifier (lazy-loaded from disk)
_svm_model: Optional[object] = None
_label_map: Optional[Dict[int, str]] = None   # int index  name


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def recognize_user(
    embedding: np.ndarray,
    prototype_embeddings: Dict[str, List[np.ndarray]],
) -> Dict[str, Any]:
    """
    Identify the person whose *embedding* best matches the enrolled users.

    Parameters
    ----------
    embedding : np.ndarray
        512-D L2-normalised embedding from ``generate_embedding()``.
    prototype_embeddings : dict
        Mapping of ``{username: [emb1, emb2, ...]}`` where each emb is a
        512-D np.ndarray.  Loaded from the database by the caller.

    Returns
    -------
    dict
        ``{"user": str, "confidence": float, "distance": float, "method": str}``
        ``user`` is ``"unknown"`` when recognition fails.
    """
    if embedding is None or len(prototype_embeddings) == 0:
        return {"user": "unknown", "confidence": 0.0, "distance": 9.9, "method": "none"}

    # ------------------------------------------------------------------
    # Stage 1  nearest-prototype Euclidean distance (always available)
    # ------------------------------------------------------------------
    nn_user, nn_dist = _nearest_neighbour(embedding, prototype_embeddings)

    if nn_dist > DISTANCE_THRESHOLD:
        return {
            "user":       "unknown",
            "confidence": 0.0,
            "distance":   float(nn_dist),
            "method":     "euclidean",
        }

    # ------------------------------------------------------------------
    # Stage 2  SVM for probability estimate (if model is trained)
    # ------------------------------------------------------------------
    svm, label_map = _load_classifier()
    if svm is not None and label_map is not None:
        svm_result = _svm_predict(embedding, svm, label_map)
        if svm_result["confidence"] >= SVM_PROB_THRESHOLD:
            # Both stages agree: return SVM result enriched with distance
            return {
                "user":       svm_result["user"],
                "confidence": svm_result["confidence"],
                "distance":   float(nn_dist),
                "method":     "svm+euclidean",
            }

    # SVM not available or low confidence  fall back to nearest-neighbour
    confidence = max(0.0, 1.0 - (nn_dist / DISTANCE_THRESHOLD))
    return {
        "user":       nn_user,
        "confidence": round(confidence, 4),
        "distance":   float(nn_dist),
        "method":     "euclidean",
    }


def train_classifier(
    prototype_embeddings: Dict[str, List[np.ndarray]],
) -> Dict[str, Any]:
    """
    (Re)train the SVM classifier on all enrolled user embeddings and persist
    the model to disk.

    Parameters
    ----------
    prototype_embeddings : dict
        ``{username: [emb1, emb2, ...]}``

    Returns
    -------
    dict
        ``{"ok": bool, "classes": int, "samples": int, "message": str}``
    """
    global _svm_model, _label_map

    from sklearn.svm import SVC
    from sklearn.preprocessing import LabelEncoder
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import Normalizer

    # Build flat X, y arrays
    X, y_names = [], []
    for name, embs in prototype_embeddings.items():
        for emb in embs:
            X.append(emb)
            y_names.append(name)

    if len(X) < 2:
        return {"ok": False, "classes": 0, "samples": len(X),
                "message": "Need at least 2 samples to train."}

    unique_labels = set(y_names)
    if len(unique_labels) < 2:
        return {
            "ok": False, "classes": len(unique_labels), "samples": len(X),
            "message": (
                "Need at least 2 different enrolled users to train the SVM. "
                "Only 1 user found. Nearest-neighbour recognition will still work."
            ),
        }

    X = np.array(X, dtype=np.float32)

    le = LabelEncoder()
    y  = le.fit_transform(y_names)

    # SVM with RBF kernel; probability=True enables predict_proba
    pipeline = Pipeline([
        ("norm",  Normalizer()),
        ("svm",   SVC(kernel="rbf", probability=True, C=5.0, gamma="scale")),
    ])
    pipeline.fit(X, y)

    # Build intname label map
    label_map = {int(i): str(name) for i, name in enumerate(le.classes_)}

    # Persist to disk
    _MODEL_DIR.mkdir(parents=True, exist_ok=True)
    with open(_SVM_PATH, "wb") as f:
        pickle.dump(pipeline, f)
    with open(_LABELS_PATH, "w") as f:
        json.dump(label_map, f, indent=2)

    # Update cache
    _svm_model  = pipeline
    _label_map  = label_map

    logger.info("SVM trained: %d classes, %d samples.", len(label_map), len(X))
    return {
        "ok":      True,
        "classes": len(label_map),
        "samples": len(X),
        "message": "Classifier trained successfully.",
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _nearest_neighbour(
    embedding: np.ndarray,
    prototypes: Dict[str, List[np.ndarray]],
) -> Tuple[str, float]:
    """Return (username, euclidean_distance) for the closest stored embedding."""
    best_name = "unknown"
    best_dist = float("inf")

    for name, embs in prototypes.items():
        for proto in embs:
            dist = float(np.linalg.norm(embedding - proto))
            if dist < best_dist:
                best_dist = dist
                best_name = name

    return best_name, best_dist


def _svm_predict(
    embedding: np.ndarray,
    svm,
    label_map: Dict[int, str],
) -> Dict[str, Any]:
    """Run SVM prediction; return name and probability."""
    probs  = svm.predict_proba(embedding.reshape(1, -1))[0]
    idx    = int(np.argmax(probs))
    return {
        "user":       label_map.get(idx, "unknown"),
        "confidence": float(probs[idx]),
    }


def _load_classifier() -> Tuple[Optional[object], Optional[Dict[int, str]]]:
    """Load the SVM pickle from disk (cached after first load)."""
    global _svm_model, _label_map

    if _svm_model is not None:
        return _svm_model, _label_map

    if _SVM_PATH.exists() and _LABELS_PATH.exists():
        try:
            with open(_SVM_PATH, "rb") as f:
                _svm_model = pickle.load(f)
            with open(_LABELS_PATH, "r") as f:
                raw = json.load(f)
                _label_map = {int(k): v for k, v in raw.items()}
            logger.info("SVM classifier loaded from disk.")
        except Exception as exc:
            logger.warning("Could not load SVM classifier: %s", exc)
            _svm_model = None
            _label_map = None

    return _svm_model, _label_map
