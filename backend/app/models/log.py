from app.models import db


class AccessLog(db.Model):
    """
    Immutable audit record for every authentication attempt.

    Fields
    ------
    timestamp   : ISO-8601 datetime string.
    status      : "granted" or "denied".
    user        : Recognised username, or "unknown".
    liveness    : Whether the liveness challenge was passed.
    confidence  : Recognition confidence score (0-100 integer %).
    detail      : Human-readable explanation of the outcome.
    ip_address  : Optional client IP for audit trail.
    """
    id          = db.Column(db.Integer,  primary_key=True)
    timestamp   = db.Column(db.String(50), nullable=False)
    status      = db.Column(db.String(20), nullable=False, default="denied")
    user        = db.Column(db.String(100), nullable=True,  default="unknown")
    liveness    = db.Column(db.Boolean,     nullable=False, default=False)
    confidence  = db.Column(db.Float,       nullable=False, default=0.0)
    detail      = db.Column(db.String(255), nullable=True)
    ip_address  = db.Column(db.String(50),  nullable=True)

    # Keep legacy ``result`` as a computed alias so the existing UI
    # dashboard does not need changes.
    @property
    def result(self) -> str:
        """Legacy alias: maps status to 'Authorized' / 'Unauthorized'."""
        return "Authorized" if self.status == "granted" else "Unauthorized"

    def to_dict(self) -> dict:
        return {
            "id":         self.id,
            "timestamp":  self.timestamp,
            "status":     self.status,
            "result":     self.result,          # legacy field kept for UI
            "user":       self.user,
            "liveness":   self.liveness,
            "confidence": round(self.confidence * 100, 1),  # 0-100 %
            "detail":     self.detail,
            "ip_address": self.ip_address,
        }
