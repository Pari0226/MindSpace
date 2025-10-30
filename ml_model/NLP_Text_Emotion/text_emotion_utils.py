import joblib
import numpy as np
import os

"""Utility functions to load the emotion classifier and expose helpers."""

# Build absolute path to the model relative to this file
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, 'models', 'emotion_classifier_pipe_lr_03_jan_2022.pkl')

# Load model with absolute path and safe fallbacks
try:
    pipe_lr = joblib.load(model_path)
    print(f"✅ Model loaded from: {model_path}")
except Exception as e:
    print(f"❌ ERROR loading model from {model_path}: {e}")
    pipe_lr = None

def predict_emotions(text):
    if pipe_lr is None:
        return 'neutral'  # fallback if model not available
    return pipe_lr.predict([text])[0]

def get_prediction_proba(text):
    if pipe_lr is None:
        # fallback: one-hot-like neutral-ish vector length 10
        return np.array([1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    return pipe_lr.predict_proba([text])[0]

emotions_emoji_dict = {
    "anger": "😠", "disgust": "🤮", "fear": "😨😱", "happy": "🤗",
    "joy": "😂", "neutral": "😐", "sad": "😔", "sadness": "😔",
    "shame": "😳", "surprise": "😮"
}
