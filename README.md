# 🌈 MindSpace – Your Personalized Mood Tracking Companion

**MindSpace** is a full-stack intelligent web application designed to help users track, visualize, and understand their emotional and mental well-being. From logging moods manually to detecting emotions through face, text, and voice, it creates a powerful and personal emotional space — like **Flo**, but for **everyone**.

---

## 🎥 Demo

Want to see MindSpace in action?  
▶️ [Click here to watch the demo video](https://drive.google.com/file/d/14vbV7SeYCIE2FHFp8mXIy6L0zlDEGPzK/view)

---

## ✨ Features

### 🎨 User Interface
- 🔥 Splash screen with logo and tagline
- 🔐 Login and registration pages with authentication
- 🧾 About page for app overview
- 📊 Dashboard with:
  - Mood logging dropdown
  - Calendar-based mood history
  - Pie chart visualization of emotions

### 🤖 Emotion Detection (ML-Powered)
- 🎥 **Facial Emotion Detection** – detects emotional state using webcam
- 📝 **Text Emotion Analysis** – identifies emotions from user-typed input
- 🎙️ **Voice Emotion Analysis** – detects emotion based on speech audio

### 📚 Mood Tracking System
- 🗓️ Log current, past, or monthly moods via dropdown
- 💾 Stores structured data into `mood.db` without mixing with emotion detection logs
- 📈 Displays visual trends over time for self-reflection

### 🧘 Mental Wellness Add-ons
- 🧩 Tips and suggestions based on mood data
- 📋 Survey form for user input to refine recommendations
- 🧠 Symptom explanations and emotional insight cards

### 💬 MoodMan – AI Chatbot (In Progress)
- Empathetic AI chatbot offering a safe space to talk
- Gives emotional support and self-help ideas instantly

---

## 📊 Tech Stack

| Frontend                         | Backend       | ML/AI Models                              | Database |
|----------------------------------|---------------|--------------------------------------------|----------|
| HTML, CSS, Tailwind, JavaScript | Flask (Python) | CNN, TF-IDF + SVM, MFCC + Random Forest   | SQLite   |

---

## 🗂️ Project Structure

├── static/
│ └── css, js, images
├── templates/
│ └── login.html, dashboard.html, etc.
├── models.py
├── app.py
├── ml_models/
│ └── face_model.h5, text_model.pkl, voice_model.pkl
├── database/
│ └── app.db, mood.db
└── README.md

---

## 🚀 Getting Started

### 1️⃣ Clone the repository

```bash
git clone https://github.com/yourusername/-mood-tracker-app.git
cd mood-tracker-app
 Set up virtual environment
bash
Copy
Edit
python -m venv venv
venv\Scripts\activate  # Windows
3️⃣ Install dependencies
bash
Copy
Edit
pip install -r requirements.txt
4️⃣ Run the app
bash
Copy
Edit
python app.py
Then go to:
📍 http://localhost:5000

🧠 Machine Learning Models Used
Task	Input	Model Type
Emotion Detection	Face	CNN (.h5 model)
Emotion Detection	Text	TF-IDF + SVM (.pkl)
Emotion Detection	Voice	MFCC + Random Forest (.pkl)

💡 Inspiration
This app is inspired by the belief that emotional wellness should be trackable, understandable, and improvable just like physical health.
Drawing from apps like Flo, Wysa, and Daylio, MindSpace provides a gender-neutral, inclusive space for anyone to care for their mental health.

👩‍💻 Author
Pari Singh
B.Tech CSE | Developer 
