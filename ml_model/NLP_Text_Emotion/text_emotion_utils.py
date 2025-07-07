import joblib
import numpy as np

# Load model (adjust path as needed if this fails)
pipe_lr = joblib.load('ml_model/NLP_Text_Emotion/models/emotion_classifier_pipe_lr_03_jan_2022.pkl')

def predict_emotions(text):
    return pipe_lr.predict([text])[0]

def get_prediction_proba(text):
    return pipe_lr.predict_proba([text])[0]

emotions_emoji_dict = {
    "anger": "😠", "disgust": "🤮", "fear": "😨😱", "happy": "🤗",
    "joy": "😂", "neutral": "😐", "sad": "😔", "sadness": "😔",
    "shame": "😳", "surprise": "😮"
}
