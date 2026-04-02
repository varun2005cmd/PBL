# =============================================================================
# app/mockml.py
# LEGACY fallback only  kept for the /scan endpoint when no image is uploaded.
# All real authentication now runs through app/ml/pipeline.py.
# =============================================================================

import random


def mock_face_scan() -> dict:
    """
    Return a random scan result.
    Used only when /scan is called without an image (old UI behaviour).
    """
    return {
        "result":     random.choice(["Authorized", "Unauthorized"]),
        "confidence": random.randint(75, 99),
    }

