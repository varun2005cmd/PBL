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


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)))
    except Exception:
        return float(default)


# Optimized Borderline Thresholds
DISTANCE_THRESHOLD = _env_float("DISTANCE_THRESHOLD", 0.95)
SVM_PROB_THRESHOLD = _env_float("SVM_PROB_THRESHOLD", 0.45)
AMBIGUITY_MARGIN   = _env_float("AMBIGUITY_MARGIN", 0.03)
STRONG_MATCH_DIST  = _env_float("STRONG_MATCH_DIST", 0.75)
NN_TOPK            = max(1, int(_env_float("NN_TOPK", 3)))
ADAPTIVE_SIGMA     = _env_float("ADAPTIVE_SIGMA", 2.2)
ADAPTIVE_MAX_RELAX = _env_float("ADAPTIVE_MAX_RELAX", 0.10)
SVM_RESCUE_THRESHOLD = _env_float("SVM_RESCUE_THRESHOLD", 0.55)
SVM_RESCUE_MARGIN    = _env_float("SVM_RESCUE_MARGIN", 0.08)

# Guard globals with a reentrant lock
_model_lock = threading.RLock()
_svm_model: Optional[object] = None
_label_map: Optional[Dict[int, str]] = None


def _clear_classifier_artifacts() -> None:
    """Remove persisted SVM files and reset the in-memory cache."""
    global _svm_model, _label_map
    _svm_model = None
    _label_map = None
    try:
        _SVM_PATH.unlink(missing_ok=True)
    except Exception:
        pass
    try:
        _LABELS_PATH.unlink(missing_ok=True)
    except Exception:
        pass


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

    # Normalize embedding for stable distance comparisons
    try:
        embedding = np.asarray(embedding, dtype=np.float32).reshape(-1)
        n = float(np.linalg.norm(embedding))
        if not np.isfinite(n) or n < 1e-8:
            return {"user": "unknown", "confidence": 0.0, "distance": 9.9, "method": "none"}
        embedding = embedding / n
    except Exception:
        return {"user": "unknown", "confidence": 0.0, "distance": 9.9, "method": "none"}

    # --- Stage 1: Robust Distance (Source of Truth) ---
    # Compute per-user distance as min(distance-to-centroid, distance-to-closest-sample)
    # then apply open-set gates: absolute threshold + ambiguity margin.
    nn_user, nn_dist, nn_thr, second_user, second_dist = _nearest_neighbour(embedding, prototype_embeddings)

    distance_gated = nn_dist > nn_thr

    # If the best match is extremely strong, allow it even if the runner-up is close.
    if nn_dist > STRONG_MATCH_DIST and second_user != "unknown" and (second_dist - nn_dist) < AMBIGUITY_MARGIN:
        return {
            "user":       "unknown",
            "confidence": 0.0,
            "distance":   float(nn_dist),
            "method":     "ambiguity_gate",
        }

    # --- Stage 2: SVM Probability ---
    # Wrap in lock to prevent training from swapping the model mid-prediction
    with _model_lock:
        svm, label_map = _load_classifier_locked()
        if svm is not None and label_map is not None:
            try:
                svm_result = _svm_predict(embedding, svm, label_map)
                svm_user = str(svm_result.get("user") or "unknown")
                svm_conf = float(svm_result.get("confidence") or 0.0)

                # Borderline rescue for enrolled users:
                # if distance is only slightly above threshold and SVM strongly
                # agrees with nearest-neighbour identity, allow the match.
                if (
                    distance_gated
                    and nn_dist <= (nn_thr + max(0.0, SVM_RESCUE_MARGIN))
                    and svm_conf >= SVM_RESCUE_THRESHOLD
                    and svm_user == nn_user
                ):
                    return {
                        "user":       nn_user,
                        "confidence": svm_conf,
                        "distance":   float(nn_dist),
                        "method":     "svm_rescue",
                    }

                if (
                    svm_conf >= SVM_PROB_THRESHOLD
                    and svm_user == nn_user
                ):
                    return {
                        "user":       nn_user,
                        "confidence": svm_conf,
                        "distance":   float(nn_dist),
                        "method":     "svm+euclidean",
                    }
            except Exception as exc:
                logger.error("SVM prediction failed: %s", exc)

    if distance_gated:
        return {
            "user":       "unknown",
            "confidence": 0.0,
            "distance":   float(nn_dist),
            "method":     "distance_gate",
        }

    # Fallback to nearest-neighbour if SVM fails or is low confidence
    confidence = max(0.0, 1.0 - (nn_dist / max(nn_thr, 1e-6)))
    return {
        "user":       nn_user,
        "confidence": round(confidence, 4),
        "distance":   float(nn_dist),
        "method":     "distance",
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
        _clear_classifier_artifacts()
        return {"ok": False, "classes": 0, "samples": len(X), "message": "Need at least 2 samples to train."}

    unique_labels = set(y_names)
    if len(unique_labels) < 2:
        _clear_classifier_artifacts()
        return {"ok": False, "classes": len(unique_labels), "message": "Need at least 2 different users."}

    try:
        X_arr = np.array(X, dtype=np.float32)
        le = LabelEncoder()
        y  = le.fit_transform(y_names)

        # Build model in memory first (off-lock)
        from sklearn.model_selection import GridSearchCV
        
        base_svm = SVC(kernel="linear", probability=True, class_weight="balanced")
        param_grid = {"C": [0.1, 1.0, 5.0, 10.0, 50.0]}
        
        # Optimize C for the best margin based on the current user dataset
        cv_count = min(3, len(unique_labels))
        grid = GridSearchCV(base_svm, param_grid, cv=cv_count, scoring="accuracy", n_jobs=1)
        
        pipeline = Pipeline([
            ("norm",  Normalizer()),
            ("svm",   grid),
        ])
        pipeline.fit(X_arr, y)
        
        # Best C found
        best_c = grid.best_params_["C"]
        logger.info("SVM optimized with best C=%.1f", best_c)

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

def _nearest_neighbour(
    embedding: np.ndarray,
    prototypes: Dict[str, List[np.ndarray]],
) -> Tuple[str, float, float, str, float]:
    """Return (best_user, best_dist, best_threshold, second_user, second_dist)."""

    best_name, best_dist = "unknown", float("inf")
    best_thr = DISTANCE_THRESHOLD
    second_name, second_dist = "unknown", float("inf")

    for name, embs in (prototypes or {}).items():
        if not embs:
            continue

        # Normalize and drop invalid entries
        cleaned: List[np.ndarray] = []
        for e in embs:
            try:
                v = np.asarray(e, dtype=np.float32).reshape(-1)
                n = float(np.linalg.norm(v))
                if not np.isfinite(n) or n < 1e-8:
                    continue
                cleaned.append(v / n)
            except Exception:
                continue

        if not cleaned:
            continue

        arr = np.stack(cleaned, axis=0)  # (k, 512)
        centroid = np.mean(arr, axis=0)
        cn = float(np.linalg.norm(centroid))
        if np.isfinite(cn) and cn > 1e-8:
            centroid = centroid / cn
            d_centroid = float(np.linalg.norm(embedding - centroid))
        else:
            d_centroid = float("inf")

        d_samples = np.linalg.norm(arr - embedding.reshape(1, -1), axis=1).astype(np.float32)
        d_min = float(np.min(d_samples)) if d_samples.size else float("inf")

        k = min(NN_TOPK, int(d_samples.size))
        if k > 0:
            topk = np.partition(d_samples, k - 1)[:k]
            d_topk = float(np.mean(topk))
        else:
            d_topk = d_min

        # Blend nearest and top-k means for robustness to single noisy templates.
        d_blend = 0.65 * d_min + 0.35 * d_topk
        d_user = min(d_centroid, float(d_blend))
        user_thr = _adaptive_user_threshold(arr)
        if d_user < best_dist:
            second_name, second_dist = best_name, best_dist
            best_name, best_dist = name, d_user
            best_thr = user_thr
        elif d_user < second_dist:
            second_name, second_dist = name, d_user

    return str(best_name), float(best_dist), float(best_thr), str(second_name), float(second_dist)


def _adaptive_user_threshold(user_embs: np.ndarray) -> float:
    """
    Derive a per-user threshold from enrollment spread.
    Wider natural variation gets a modestly relaxed threshold.
    """
    try:
        if user_embs.size == 0:
            return float(DISTANCE_THRESHOLD)

        centroid = np.mean(user_embs, axis=0)
        cn = float(np.linalg.norm(centroid))
        if not np.isfinite(cn) or cn <= 1e-8:
            return float(DISTANCE_THRESHOLD)
        centroid = centroid / cn

        intra = np.linalg.norm(user_embs - centroid.reshape(1, -1), axis=1).astype(np.float32)
        mu = float(np.mean(intra))
        sigma = float(np.std(intra))
        adaptive = mu + ADAPTIVE_SIGMA * sigma + 0.02
        max_allowed = DISTANCE_THRESHOLD + max(0.0, ADAPTIVE_MAX_RELAX)
        return float(max(DISTANCE_THRESHOLD, min(adaptive, max_allowed)))
    except Exception:
        return float(DISTANCE_THRESHOLD)


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
            _clear_classifier_artifacts()
    return _svm_model, _label_map
