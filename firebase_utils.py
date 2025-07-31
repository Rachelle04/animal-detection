# firebase_utils.py
import firebase_admin
from firebase_admin import credentials, firestore
import datetime
from firebase_admin import firestore as fs_admin

if not firebase_admin._apps:
    cred = credentials.Certificate("firebase_config.json")
    firebase_admin.initialize_app(cred)

db = firestore.client()

def save_to_firebase(animal, emotion, confidence_animal, confidence_emotion,
                     loss=0.0, accuracy=0.0, video_path=None, audio_path=None):
    """Save detection results to Firebase with proper timestamp formatting"""
    timestamp_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    try:
        doc_ref = db.collection("detections").add({
            "animal": str(animal).lower(),
            "emotion": str(emotion).lower(),
            "confidence_animal": float(confidence_animal),
            "confidence_emotion": float(confidence_emotion),
            "loss": float(loss),
            "accuracy": float(accuracy),
            "timestamp": fs_admin.SERVER_TIMESTAMP,
            "timestamp_str": timestamp_str,
            "video_file": str(video_path) if video_path else "",
            "audio_file": str(audio_path) if audio_path else ""
        })
        print(f"✅ Document written with ID: {doc_ref[1].id}")
        return True
    except Exception as e:
        print(f"❌ Error adding document: {e}")
        return False
