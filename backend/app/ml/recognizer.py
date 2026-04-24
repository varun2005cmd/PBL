# =============================================================================
# app/ml/recognizer.py  –  v2 (hardened)
#
# Fixes applied:
#   • Added _model_lock (RLock) to protect global model/label_map state
#   • recognize_user: validates input and holds lock during prediction
#   • train_classifier: holds lock during update of globals
#   • Atomic-style training: builds new model before acquiring lock to swap
#   • Path safety: ensures model_store exists before writing
# =============================================================================

from __future__ import annotations

import os
import json
import pickle
import logging
import threading
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
_MODEL_DIR  = Path(__file__).parent / "model_store"
_SVM_PATH   = _MODEL_DIR / "svm_classifier.pkl"
_LABELS_PATH = _MODEL_DIR / "label_map.json"

DISTANCE_THRESHOLD = 0.9
SVM_PROB_THRESHOLD = 0.55

# Guard globals with a reentrant lock
_model_lock = threading.RLock()
_svm_model: Optional[object] = None
_label_map: Optional[Dict[int, str]] = None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def recognize_user(
    embedding: np.ndarray,
    prototype_embeddings: Dict[str, List[np.ndarray]],
) -> Dict[str, Any]:
    """
    Identify the person whose *embedding* best matches the enrolled users.
    Thread-safe; holds _model_lock during Stage 2.
    """
    if embedding is None or not isinstance(embedding, np.ndarray):
        return {"user": "unknown", "confidence": 0.0, "distance": 9.9, "method": "none"}

    if not prototype_embeddings:
        return {"user": "unknown", "confidence": 0.0, "distance": 9.9, "method": "none"}

    # --- Stage 1: Euclidean Distance (Source of Truth) ---
    nn_user, nn_dist = _nearest_neighbour(embedding, prototype_embeddings)

    if nn_dist > DISTANCE_THRESHOLD:
        return {
            "user":       "unknown",
            "confidence": 0.0,
            "distance":   float(nn_dist),
            "method":     "euclidean",
        }

    # --- Stage 2: SVM Probability ---
    # Wrap in lock to prevent training from swapping the model mid-prediction
    with _model_lock:
        svm, label_map = _load_classifier_locked()
        if svm is not None and label_map is not None:
            try:
                svm_result = _svm_predict(embedding, svm, label_map)
                if (
                    float(svm_result.get("confidence") or 0.0) >= SVM_PROB_THRESHOLD
                    and str(svm_result.get("user") or "unknown") == nn_user
                ):
                    return {
                        "user":       nn_user,
                        "confidence": float(svm_result.get("confidence") or 0.0),
                        "distance":   float(nn_dist),
                        "method":     "svm+euclidean",
                    }
            except Exception as exc:
                logger.error("SVM prediction failed: %s", exc)

    # Fallback to nearest-neighbour if SVM fails or is low confidence
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
    (Re)train the SVM classifier and persist to disk.
    Thread-safe; acquires _model_lock only for the final swap.
    """
    try:
        from sklearn.svm import SVC
        from sklearn.preprocessing import LabelEncoder
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import Normalizer
    except ImportError:
        return {"ok": False, "message": "scikit-learn not installed"}

    X, y_names = [], []
    for name, embs in prototype_embeddings.items():
        for emb in embs:
            X.append(emb)
            y_names.append(name)

    if len(X) < 2:
        return {"ok": False, "classes": 0, "samples": len(X), "message": "Need at least 2 samples to train."}

    unique_labels = set(y_names)
    if len(unique_labels) < 2:
        return {"ok": False, "classes": len(unique_labels), "message": "Need at least 2 different users."}

    try:
        X_arr = np.array(X, dtype=np.float32)
        le = LabelEncoder()
        y  = le.fit_transform(y_names)

        # Build model in memory first (off-lock)
        pipeline = Pipeline([
            ("norm",  Normalizer()),
            ("svm",   SVC(kernel="rbf", probability=True, C=5.0, gamma="scale")),
        ])
        pipeline.fit(X_arr, y)

        new_label_map = {int(i): str(name) for i, name in enumerate(le.classes_)}

        # Persist to disk
        _MODEL_DIR.mkdir(parents=True, exist_ok=True)
        with open(_SVM_PATH, "wb") as f:
            pickle.dump(pipeline, f)
        with open(_LABELS_PATH, "w") as f:
            json.dump(new_label_map, f, indent=2)

        # Atomic swap under lock
        global _svm_model, _label_map
        with _model_lock:
            _svm_model = pipeline
            _label_map = new_label_map

        logger.info("SVM trained: %d classes, %d samples.", len(new_label_map), len(X))
        return {"ok": True, "classes": len(new_label_map), "samples": len(X)}

    except Exception as exc:
        logger.error("Training failed: %s", exc)
        return {"ok": False, "message": f"Training error: {str(exc)}"}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _nearest_neighbour(embedding: np.ndarray, prototypes: Dict[str, List[np.ndarray]]) -> Tuple[str, float]:
    best_name, best_dist = "unknown", float("inf")
    for name, embs in prototypes.items():
        for proto in embs:
            dist = float(np.linalg.norm(embedding - proto))
            if dist < best_dist:
                best_dist, best_name = dist, name
    return best_name, best_dist


def _svm_predict(embedding: np.ndarray, svm, label_map: Dict[int, str]) -> Dict[str, Any]:
    probs  = svm.predict_proba(embedding.reshape(1, -1))[0]
    idx    = int(np.argmax(probs))
    return {"user": label_map.get(idx, "unknown"), "confidence": float(probs[idx])}


def _load_classifier_locked() -> Tuple[Optional[object], Optional[Dict[int, str]]]:
    """Caller must hold _model_lock."""
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
    return _svm_model, _label_map
