import json
from app.models import db


class User(db.Model):
    """
    Registered user with stored face embeddings for recognition.

    ``embeddings_json`` holds a JSON-encoded list of 512-D float lists:
      [[0.12, -0.34, ...], [0.09, -0.41, ...], ...]   (one per enrollment frame)
    """
    id              = db.Column(db.Integer,      primary_key=True)
    name            = db.Column(db.String(100),  nullable=False, unique=True)
    active          = db.Column(db.Boolean,      default=True)
    enrolled        = db.Column(db.Boolean,      default=False)
    # Serialised list-of-lists; each inner list is a 512-D float embedding
    embeddings_json = db.Column(db.Text,         nullable=True)

    # ----------------------------------------------------------------
    # Embedding helpers
    # ----------------------------------------------------------------
    def set_embeddings(self, embeddings: list) -> None:
        """Persist a list of 512-D float lists."""
        self.embeddings_json = json.dumps(embeddings)
        self.enrolled = bool(embeddings)

    def get_embeddings(self) -> list:
        """Return stored embeddings as a list of Python lists (or [])."""
        if not self.embeddings_json:
            return []
        return json.loads(self.embeddings_json)

    def get_np_embeddings(self):
        """Return stored embeddings as a list of numpy float32 arrays."""
        import numpy as np
        return [
            np.array(e, dtype=np.float32)
            for e in self.get_embeddings()
        ]

    def to_dict(self) -> dict:
        return {
            "id":         self.id,
            "name":       self.name,
            "active":     self.active,
            "enrolled":   self.enrolled,
            # Fields expected by the frontend UserManagement page:
            "status":     "enabled" if self.active else "disabled",
            "userId":     f"usr_{self.id:05d}",
            "role":       "User",
            "lastAccess": None,   # not tracked per-user; logs hold per-attempt data
        }
