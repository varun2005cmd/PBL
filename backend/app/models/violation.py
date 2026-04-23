from app.models import db


class ViolationImage(db.Model):
    """Evidence image captured when repeat-person policy denies access."""

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)
    timestamp = db.Column(db.String(50), nullable=False)
    image_path = db.Column(db.String(255), nullable=False)
    group_id = db.Column(db.String(80), nullable=False, index=True)

    user = db.relationship("User", backref=db.backref("violation_images", lazy=True))

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "user_id": self.user_id,
            "userName": self.user.name if self.user else "unknown",
            "timestamp": self.timestamp,
            "imagePath": self.image_path,
            "imageUrl": f"/violations/image/{self.image_path}",
            "groupId": self.group_id,
        }
