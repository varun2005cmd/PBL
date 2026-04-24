from datetime import datetime
from app.models import db

class AccessLog(db.Model):
    __tablename__ = "access_logs"

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    timestamp = db.Column(db.String(50), nullable=False)
    status = db.Column(db.String(20), nullable=False)
    user = db.Column(db.String(100), nullable=True)
    liveness = db.Column(db.Boolean, default=False)
    confidence = db.Column(db.Float, default=0.0)
    detail = db.Column(db.String(200), nullable=True)
    ip_address = db.Column(db.String(50), nullable=True)

    def to_dict(self):
        return {
            "id": self.id,
            "timestamp": self.timestamp,
            "status": self.status,
            "user": self.user,
            "liveness": self.liveness,
            "confidence": self.confidence,
            "detail": self.detail,
            "ip_address": self.ip_address,
        }
