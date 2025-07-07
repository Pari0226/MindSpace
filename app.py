from flask import Flask, render_template, request, redirect, flash, session, Response
from flask_sqlalchemy import SQLAlchemy
from datetime import date, timedelta
from datetime import datetime
from collections import Counter
import calendar
import cv2
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import img_to_array
import joblib
import os
import pandas as pd

# Initialize app
app = Flask(__name__)
app.secret_key = 'supersecretkey'

# Configure database
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# Define models
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(150))
    age = db.Column(db.Integer)
    email = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(150), nullable=False)

class Mood(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    mood = db.Column(db.String(50), nullable=False)
    date = db.Column(db.Date, default=date.today)

class TextEmotion(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    text = db.Column(db.Text, nullable=False)
    emotion = db.Column(db.String(50), nullable=False)
    date = db.Column(db.Date, default=date.today)

# Load models
face_classifier = cv2.CascadeClassifier("C:/Users/Lenovo/Desktop/mood tracker app/ml_model/Emotion_Dectector/haarcascade_frontalface_default.xml")
classifier = load_model("C:/Users/Lenovo/Desktop/mood tracker app/ml_model/Emotion_Dectector/model.h5")
nlp_model_path = "C:/Users/Lenovo/Desktop/mood tracker app/ml_model/NLP_Text_Emotion/models/emotion_classifier_pipe_lr_03_jan_2022.pkl"
nlp_model = joblib.load(nlp_model_path)
emotion_labels = ['Angry','Disgust','Fear','Happy','Neutral','Sad','Surprise']

# NLP utilities
from ml_model.NLP_Text_Emotion.text_emotion_utils import (
    predict_emotions, get_prediction_proba, emotions_emoji_dict
)

# Routes
@app.route('/')
def splash():
    return render_template('splash.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        user = User.query.filter_by(email=email).first()
        if user and user.password == password:
            session['user'] = user.name
            flash('Login successful!', 'success')
            return redirect('/dashboard')
        else:
            flash('User does not exist or password is incorrect.', 'error')
            return redirect('/login')
    return render_template('login.html')

@app.route('/register', methods=['POST'])
def register():
    name = request.form['name']
    age = request.form['age']
    email = request.form['email']
    password = request.form['password']

    if User.query.filter_by(email=email).first():
        flash('Email already registered. Please login.', 'error')
        return redirect('/login')

    new_user = User(name=name, age=age, email=email, password=password)
    db.session.add(new_user)
    db.session.commit()
    flash('User registered successfully!', 'success')
    return redirect('/login')

@app.route('/logout')
def logout():
    session.pop('user', None)
    flash('Logged out successfully.', 'success')
    return redirect('/login')

@app.route('/about')
def about():
    return render_template('about.html')


# Helper: Calculate streaks
def calculate_streaks(dates):
    if not dates:
        return 0, 0

    dates = sorted(set(dates))
    longest_streak = current_streak = 1
    today = date.today()
    
    temp_streak = 1
    for i in range(1, len(dates)):
        if (dates[i] - dates[i-1]).days == 1:
            temp_streak += 1
        else:
            longest_streak = max(longest_streak, temp_streak)
            temp_streak = 1
    longest_streak = max(longest_streak, temp_streak)

    # Current streak check
    current_streak = 1
    for i in range(len(dates)-1, 0, -1):
        if (dates[i] - dates[i-1]).days == 1:
            current_streak += 1
        elif dates[i] == today:
            continue
        else:
            break

    # If last mood wasn't today or yesterday, reset current_streak
    if (today - dates[-1]).days > 1:
        current_streak = 0

    return current_streak, longest_streak

@app.route('/dashboard')
def dashboard():
    if 'user' not in session:
        flash('Please log in first.', 'error')
        return redirect('/login')

    user = User.query.filter_by(name=session['user']).first()
    if not user:
        flash('User not found.', 'error')
        return redirect('/login')

    year = request.args.get('year', default=date.today().year, type=int)
    month = request.args.get('month', default=date.today().month, type=int)

    today = date.today()
    first_day = date(year, month, 1)
    last_day = date(year, month, calendar.monthrange(year, month)[1])
    start_day = (first_day.weekday() + 1) % 7
    days_in_month = (last_day - first_day).days + 1

    moods = Mood.query.filter_by(user_id=user.id).order_by(Mood.date.desc()).all()
    texts = TextEmotion.query.filter_by(user_id=user.id).order_by(TextEmotion.date.desc()).all()

    calendar_data = {}
    for mood in moods:
        if first_day <= mood.date <= last_day:
            calendar_data[mood.date.day] = mood.mood

    mood_dates = [m.date for m in moods]
    current_streak, longest_streak = calculate_streaks(mood_dates)

    mood_count = dict(Counter([m.mood for m in moods]))

    prev_month = month - 1 if month > 1 else 12
    prev_year = year if month > 1 else year - 1
    next_month = month + 1 if month < 12 else 1
    next_year = year if month < 12 else year + 1

    return render_template('dashboard.html',
                       user=session['user'],
                       moods=moods,
                       texts=texts,
                       calendar_data=calendar_data,
                       today=today,
                       year=year,
                       month=month,
                       start_day=start_day,
                       days_in_month=days_in_month,
                       prev_month=prev_month,
                       prev_year=prev_year,
                       next_month=next_month,
                       next_year=next_year,
                       mood_count=mood_count,
                       current_streak=current_streak,
                       longest_streak=longest_streak,
                       calendar=calendar)

@app.route('/log_mood', methods=['POST'])
def log_mood():
    if 'user' not in session:
        flash('Please log in first.', 'error')
        return redirect('/login')

    mood_value = request.form.get('mood')
    mood_date = request.form.get('mood_date')

    if not mood_value or not mood_date:
        flash('Please select both a mood and a date.', 'error')
        return redirect('/dashboard')

    user = User.query.filter_by(name=session['user']).first()
    mood_date = date.fromisoformat(mood_date)
    existing_mood = Mood.query.filter_by(user_id=user.id, date=mood_date).first()

    if existing_mood:
        existing_mood.mood = mood_value
        flash(f'Updated your mood for {mood_date.strftime("%B %d")}.', 'success')
    else:
        db.session.add(Mood(user_id=user.id, mood=mood_value, date=mood_date))
        flash(f'Mood saved for {mood_date.strftime("%B %d")}.', 'success')

    db.session.commit()
    return redirect('/dashboard')

@app.route('/face_emotion')
def detect_emotion():
    return render_template('detect_emotion.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

def generate_frames():
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            roi_gray = cv2.resize(roi_gray, (48, 48))
            if np.sum(roi_gray) != 0:
                roi = roi_gray.astype("float") / 255.0
                roi = img_to_array(roi)
                roi = np.expand_dims(roi, axis=0)
                prediction = classifier.predict(roi)[0]
                label = emotion_labels[prediction.argmax()]
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    cap.release()

@app.route('/text_emotion', methods=['GET', 'POST'])
def text_emotion():
    if 'user' not in session:
        flash('Please log in first.', 'error')
        return redirect('/login')

    if request.method == 'POST':
        text = request.form['text']
        if not text:
            flash("Please enter some text.", "error")
            return redirect('/text_emotion')

        emotion = predict_emotions(text)
        proba = get_prediction_proba(text)
        confidence = round(np.max(proba) * 100, 2)
        emoji = emotions_emoji_dict.get(emotion, '')

        user = User.query.filter_by(name=session['user']).first()
        db.session.add(TextEmotion(user_id=user.id, text=text, emotion=emotion))
        db.session.commit()

        return render_template('text_emotion_result.html',
                               text=text,
                               emotion=emotion,
                               emoji=emoji,
                               confidence=confidence,
                               probas=dict(zip(nlp_model.classes_, map(float, proba))))
    return render_template('text_emotion.html')

# Run app
if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)
