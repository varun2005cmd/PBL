import sys
import os

# Add backend to path
sys.path.append(os.path.join(os.getcwd(), "backend"))

from app import create_app
from app.models import db
from app.models.user import User
from app.ml.recognizer import recognize_user
import numpy as np

def validate():
    app = create_app()
    with app.app_context():
        user = User.query.filter_by(name="Palash").first()
        if not user:
            print("ERROR: User 'Palash' not found in database.")
            return

        print(f"Found user: {user.name}")
        print(f"Enrolled: {user.enrolled}")
        print(f"Active: {user.active}")
        
        embeddings = user.get_np_embeddings()
        if not embeddings:
            print("ERROR: No embeddings found for 'Palash'. Please enroll them first.")
            return
            
        print(f"Number of embeddings: {len(embeddings)}")
        
        # Test recognition logic
        # We take the first embedding and try to recognize it
        test_emb = embeddings[0]
        
        # Mock prototype dict
        prototypes = {user.name: embeddings}
        
        result = recognize_user(test_emb, prototypes)
        print("\n--- Recognition Test Results ---")
        print(f"Detected User: {result.get('user')}")
        print(f"Confidence: {result.get('confidence')}")
        print(f"Distance: {result.get('distance')}")
        print(f"Method: {result.get('method')}")
        
        if result.get('user') == "Palash":
            print("\nSUCCESS: The model correctly identified Palash's embedding!")
        else:
            print("\nFAILURE: The model failed to identify the user's own embedding.")

if __name__ == "__main__":
    validate()
