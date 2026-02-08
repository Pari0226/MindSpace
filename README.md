# 🌈 MindSpace – AI Mood Companion with 7-Days Mood Forecasting

### 🎥 Demo Video  
Want to see MindSpace in action?  
▶️ [Click here to watch the demo video](https://drive.google.com/file/d/12DqdVPHE6poTWr1ui2zGg7H_MiFS_F8t/view)

MindSpace is a mental wellness platform that helps users track emotions, reflect through journaling, and receive **AI-powered 7-day mood forecasts**.  
It combines **multi-modal emotion detection**, analytics, and personalized insights to help users understand their mental patterns over time.

---

## 🎯 What MindSpace Offers

- Daily mood logging with **intensity scale (1–10)**
- Journal writing with automatic emotion extraction  
- **Facial emotion detection** via webcam  
- Calendar & analytics dashboard  
- AI-generated personalized insights  
- **7-day mood prediction** with confidence scoring

---

## ✨ Core Features

### 🧠 Multi-Modal Emotion Detection
- **Text Emotion Analysis** – TF-IDF + Linear Regression model  
- **Facial Emotion Recognition** – CNN model (7 classes: Happy, Sad, Angry, Neutral, Fear, Surprise, Disgust)  
- **Voice Sentiment** – TextBlob based sentiment analysis

### 📊 Mood Tracking & Analytics
- Intensity-based mood logging  
- Calendar visualization  
- Emotion distribution charts  
- Streak tracking & monthly overview

### 🔮 7-Day Forecast Engine
Predictions are generated using:
- Baseline from last 7 entries  
- Weekday behavioral patterns  
- Polynomial trend slope  
- Trigger keyword weights (-0.4 to +0.3)  
- Recent momentum (70/30 blend)  
- **Confidence score: 0.3 – 0.95**

### 💡 Personalized Insights
- Best & worst day detection  
- Trigger identification from journals  
- Consistency bonus logic  
- Natural language reasoning for predictions

### 🔐 Secure System
- **Argon2 password hashing**  
- User authentication & sessions  
- SQLite database with SQLAlchemy

---

## 🛠 Tech Stack

**Backend**
- Flask  
- Flask-SQLAlchemy  
- Argon2  
- OpenCV + Keras CNN  
- Scikit-learn (TF-IDF + Linear Regression)  
- TextBlob

**Frontend**
- HTML / CSS / Tailwind  
- Vanilla JavaScript  
- Interactive dashboard UI

---

## 🧩 Database Models

- **User** – authentication & profile  
- **Mood** – daily mood + intensity  
- **JournalEntry** – text reflections  
- **TextEmotion** – extracted emotions

---

## 🚀 How to Run Locally

```bash
git clone https://github.com/yourusername/mindspace.git
cd mindspace
pip install -r requirements.txt
python app.py
```   

Open in browser → http://localhost:5000

### 📌 Prediction Logic (High Level)

- Uses last 30 entries for context  
- Baseline from recent 7 moods  
- Weekday clustering (Mon–Sun averages)  
- Trend direction via polyfit slope
- Trigger impact from journals
- Generates daily reasoning text
- Confidence clamped to 0.3–0.95

###  ⚠ Honest Notes

- Voice module is standalone
- Text model accuracy not benchmarked
- Trigger system is heuristic-based, not deep learning
- No external datasets used for training



###  💙 Built By

## Pari Singh
##  Exploring mental health through gentle technology ✨
