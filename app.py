from flask import Flask, render_template, request, redirect, flash, session, Response, jsonify
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
import re
import random
from dotenv import load_dotenv
from passlib.context import CryptContext
import pandas as pd

# Initialize app
app = Flask(__name__)
instance_dir = os.path.join(app.root_path, 'instance')
os.makedirs(instance_dir, exist_ok=True)
# Load environment variables from .env (if present)
load_dotenv()
app.secret_key = os.environ.get('APP_SECRET_KEY', 'supersecretkey')

# Configure database
basedir = os.path.abspath(os.path.dirname(__file__))
instance_path = os.path.join(basedir, 'instance')
os.makedirs(instance_path, exist_ok=True)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + os.path.join(instance_path, 'app.db').replace('\\', '/')
app.config['DB_FILE_PATH'] = os.path.join(instance_path, 'app.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# Password hashing context
pwd_context = CryptContext(schemes=["argon2"], deprecated="auto")

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
    intensity = db.Column(db.Integer, default=5)

class TextEmotion(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    text = db.Column(db.Text, nullable=False)
    emotion = db.Column(db.String(50), nullable=False)
    date = db.Column(db.Date, default=date.today)


class JournalEntry(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    entry_text = db.Column(db.Text, nullable=False)
    entry_date = db.Column(db.Date, default=date.today)
    created_at = db.Column(db.DateTime, default=datetime.now)
    mood_score = db.Column(db.Float, nullable=True)

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
        if user and pwd_context.verify(password, user.password):
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
    # Hash password before storing
    hashed_pw = pwd_context.hash(password)
    new_user = User(name=name, age=age, email=email, password=hashed_pw)
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
    intensity_raw = request.form.get('intensity')
    custom_text = (request.form.get('custom_mood') or '').strip()

    if not mood_value or not mood_date:
        flash('Please select both a mood and a date.', 'error')
        return redirect('/dashboard')

    user = User.query.filter_by(name=session['user']).first()
    mood_date = date.fromisoformat(mood_date)

    # Normalize mood (handle custom)
    normalized = mood_value.lower()
    if normalized in ['other', 'custom']:
        if custom_text:
            normalized = f"custom: {custom_text}"
        else:
            normalized = 'custom'

    # Parse intensity 1-10
    intensity_val = None
    try:
        if intensity_raw is not None and intensity_raw != '':
            iv = int(intensity_raw)
            if iv < 1: iv = 1
            if iv > 10: iv = 10
            intensity_val = iv
    except ValueError:
        intensity_val = None

    # Debug logs
    print('[log_mood] user=', user.id if user else None,
          'date=', mood_date,
          'mood_raw=', mood_value,
          'mood_normalized=', normalized,
          'intensity_raw=', intensity_raw,
          'intensity_val=', intensity_val)

    existing_mood = Mood.query.filter_by(user_id=user.id, date=mood_date).first()

    if existing_mood:
        existing_mood.mood = normalized
        existing_mood.intensity = intensity_val
        flash(f'Updated your mood for {mood_date.strftime("%B %d")}.', 'success')
    else:
        db.session.add(Mood(user_id=user.id, mood=normalized, date=mood_date, intensity=intensity_val))
        flash(f'Mood saved for {mood_date.strftime("%B %d")}.', 'success')

    db.session.commit()
    # Post-save debug
    try:
        saved = Mood.query.filter_by(user_id=user.id, date=mood_date).first()
        print('[log_mood] saved_row=', {
            'id': saved.id if saved else None,
            'mood': saved.mood if saved else None,
            'intensity': saved.intensity if saved else None,
            'date': saved.date.isoformat() if saved and saved.date else None,
        })
    except Exception as e:
        print('[log_mood] post-save read error:', e)
    return redirect('/dashboard')

@app.route('/api/mood/get', methods=['GET'])
def api_mood_get():
    if 'user' not in session:
        return jsonify({'error': 'authentication required'}), 401
    user = User.query.filter_by(name=session['user']).first()
    if not user:
        return jsonify({'error': 'user not found'}), 404
    moods = Mood.query.filter_by(user_id=user.id).order_by(Mood.date.desc()).all()
    return jsonify({'moods': [
        {
            'id': m.id,
            'date': m.date.isoformat() if m.date else None,
            'mood': m.mood,
            'intensity': m.intensity
        } for m in moods
    ]})

@app.route('/api/mood/calendar', methods=['GET'])
def api_mood_calendar():
    if 'user' not in session:
        return jsonify({'error': 'authentication required'}), 401
    user = User.query.filter_by(name=session['user']).first()
    if not user:
        return jsonify({'error': 'user not found'}), 404

    try:
        year = int(request.args.get('year', date.today().year))
        month = int(request.args.get('month', date.today().month))
        if month < 1 or month > 12:
            raise ValueError('invalid month')
    except Exception:
        return jsonify({'error': 'invalid year/month'}), 400

    month_start = date(year, month, 1)
    month_end = date(year, month, calendar.monthrange(year, month)[1])

    moods = Mood.query.filter(
        Mood.user_id == user.id,
        Mood.date >= month_start,
        Mood.date <= month_end
    ).order_by(Mood.date.asc()).all()

    return jsonify({'year': year, 'month': month, 'moods': [
        {
            'id': m.id,
            'date': m.date.isoformat() if m.date else None,
            'mood': m.mood,
            'intensity': m.intensity
        } for m in moods
    ]})

@app.route('/api/mood/emotions-this-month', methods=['GET'])
def api_emotions_this_month():
    if 'user' not in session:
        return jsonify({'error': 'authentication required'}), 401
    user = User.query.filter_by(name=session['user']).first()
    if not user:
        return jsonify({'error': 'user not found'}), 404

    today_dt = date.today()
    month_start = date(today_dt.year, today_dt.month, 1)
    month_end = date(today_dt.year, today_dt.month, calendar.monthrange(today_dt.year, today_dt.month)[1])

    # JournalEntry emotions this month (derived from text)
    journal_entries = JournalEntry.query.filter(
        JournalEntry.user_id == user.id,
        JournalEntry.entry_date >= month_start,
        JournalEntry.entry_date <= month_end
    ).all()

    def mood_to_emotion_label(mood_str: str) -> str:
        if not mood_str:
            return 'unknown'
        m = mood_str.lower()
        if any(k in m for k in ['happy','joy','excited','energized','grateful']):
            return 'joy'
        if any(k in m for k in ['calm','okay','ok','neutral']):
            return 'calm'
        if any(k in m for k in ['anxious','anxiety','fear']):
            return 'anxious'
        if any(k in m for k in ['stressed','stress','angry','anger']):
            return 'angry'
        if any(k in m for k in ['tired','fatigue','sleepy']):
            return 'tired'
        if any(k in m for k in ['sad','down','depressed']):
            return 'sad'
        return 'other'

    counts = {}
    # From journals
    for e in journal_entries:
        label, _, _, _ = extract_emotions_from_text(e.entry_text or '')
        label = (label or 'unknown').lower()
        counts[label] = counts.get(label, 0) + 1

    # Mood table entries this month
    mood_entries = Mood.query.filter(
        Mood.user_id == user.id,
        Mood.date >= month_start,
        Mood.date <= month_end
    ).all()
    for m in mood_entries:
        lbl = mood_to_emotion_label(m.mood)
        counts[lbl] = counts.get(lbl, 0) + 1

    total = sum(counts.values())
    return jsonify({'counts': counts, 'total': total, 'month': month_start.strftime('%Y-%m')})

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


def extract_emotions_from_text(text):
    """Return (emotion_label, confidence_float, mood_score_float, secondary_emotion) from text."""
    if not text:
        return None, 0.0, 0.0, None

    try:
        emotion = predict_emotions(text)
        proba = get_prediction_proba(text)

        # Top-2 emotions
        secondary_emotion = None
        try:
            top_indices = np.argsort(proba)[-2:][::-1]
            model_classes = None
            # Prefer classes_ from the app-loaded model if available, else from utils model if exposed
            if 'nlp_model' in globals() and hasattr(nlp_model, 'classes_'):
                model_classes = nlp_model.classes_
            elif 'pipe_lr' in globals() and hasattr(pipe_lr, 'classes_'):
                model_classes = pipe_lr.classes_
            if model_classes is not None and len(top_indices) > 1:
                secondary_emotion = str(model_classes[top_indices[1]])
        except Exception:
            secondary_emotion = None

        confidence = float(np.max(proba))

        # Map common labels to a -1..1 mood score and scale by confidence
        mapping = {
            'happy': 1.0, 'joy': 1.0, 'neutral': 0.0,
            'sad': -1.0, 'sadness': -1.0, 'anger': -0.9,
            'disgust': -0.9, 'fear': -0.9, 'surprise': 0.2, 'shame': -0.7
        }
        label = str(emotion).lower()
        base = mapping.get(label, 0.0)
        mood_score = base * confidence
        return emotion, confidence, float(mood_score), secondary_emotion
    except Exception:
        return None, 0.0, 0.0, None


@app.route('/api/journal/save', methods=['POST'])
def api_journal_save():
    if 'user' not in session:
        return jsonify({'error': 'authentication required'}), 401

    # accept JSON or form data
    data = request.get_json(silent=True) or request.form
    # Support both legacy 'text' and new 'content'
    text = (data.get('text') or data.get('content')) if data else None
    entry_date_str = data.get('date') if data else None
    preselected_mood = (data.get('preselected_mood') or '').strip().lower() if data else ''
    if not text:
        return jsonify({'error': 'text is required'}), 400

    user = User.query.filter_by(name=session['user']).first()
    if not user:
        return jsonify({'error': 'user not found'}), 404

    emotion, confidence, mood_score, secondary_emotion = extract_emotions_from_text(text)
    # Use preselected_mood as a hint if provided and confidence is low
    try:
        if preselected_mood and (confidence is None or float(confidence) < 0.6 or not emotion):
            # map common moods to emotion labels
            hint_map = {
                'happy': 'joy', 'joy': 'joy', 'excited': 'joy', 'calm': 'calm',
                'anxious': 'anxious', 'anxiety': 'anxious', 'stressed': 'angry', 'angry': 'angry',
                'sad': 'sad', 'tired': 'tired'
            }
            hinted = hint_map.get(preselected_mood, preselected_mood)
            emotion = hinted
            # nudge score upward for pos, downward for neg
            if hinted in ['joy','calm']:
                mood_score = max(0.0, min(1.0, (mood_score or 0.0) + 0.15))
            elif hinted in ['sad','anxious','angry','tired']:
                mood_score = max(-1.0, min(1.0, (mood_score or 0.0) - 0.15))
            confidence = max(float(confidence or 0.0), 0.65)
    except Exception:
        pass

    # Parse optional entry_date (YYYY-MM-DD) without timezone conversions
    entry_kwargs = {'user_id': user.id, 'entry_text': text, 'mood_score': mood_score}
    if entry_date_str:
        try:
            parsed_date = date.fromisoformat(entry_date_str)
            entry_kwargs['entry_date'] = parsed_date
        except Exception:
            pass

    entry = JournalEntry(**entry_kwargs)
    db.session.add(entry)
    db.session.commit()

    # Compute a quick prediction payload using existing api_predict
    prediction_payload = None
    try:
        pred_resp = api_predict()
        # api_predict returns a Flask Response
        if isinstance(pred_resp, tuple):
            pred_resp = pred_resp[0]
        pred_json = pred_resp.get_json(silent=True) if hasattr(pred_resp, 'get_json') else None
        if pred_json and isinstance(pred_json, dict) and pred_json.get('status') == 'success':
            preds = pred_json.get('predictions') or []
            if preds:
                first = preds[0]
                # map predicted_mood 0..1 to label for UX (rough categories)
                score = first.get('predicted_mood')
                emo_label = None
                try:
                    s = float(score)
                    if s >= 0.66: emo_label = 'happy'
                    elif s <= 0.33: emo_label = 'sad'
                    else: emo_label = 'calm'
                except Exception:
                    emo_label = 'calm'
                prediction_payload = {
                    'date': first.get('date'),
                    'emotion': emo_label,
                    'score': first.get('predicted_mood'),
                    'reasoning': first.get('overall_reasoning') if isinstance(pred_json, dict) else None
                }
    except Exception:
        prediction_payload = None

    return jsonify({
        'success': True,
        'status': 'saved',
        'id': entry.id,
        'saved_at': entry.created_at.isoformat() if entry.created_at else None,
        'emotion': emotion,
        'secondary_emotion': secondary_emotion,
        'confidence': float(confidence) if confidence is not None else None,
        'mood_score': float(mood_score) if mood_score is not None else None,
        'prediction': prediction_payload
    }), 201


@app.route('/api/journal/get', methods=['GET'])
def api_journal_get():
    if 'user' not in session:
        return jsonify({'error': 'authentication required'}), 401
    user = User.query.filter_by(name=session['user']).first()
    if not user:
        return jsonify({'error': 'user not found'}), 404

    entries = JournalEntry.query.filter_by(user_id=user.id).order_by(JournalEntry.created_at.desc()).all()
    result = []
    pos_strong = {"friend", "friends", "exercise", "social"}
    pos_mild = {"work", "project", "excited"}
    neg_mild = {"stress", "tired", "anxious"}
    neg_strong = {"sad", "down", "depressed"}
    for e in entries:
        emotion_label, _, _, secondary_emotion = extract_emotions_from_text(e.entry_text or '')
        words = set(re.findall(r"[a-z']+", (e.entry_text or '').lower()))
        triggered = sorted([w for w in words if w in pos_strong or w in pos_mild or w in neg_mild or w in neg_strong])
        result.append({
            'id': e.id,
            'entry_text': e.entry_text,
            'entry_date': e.entry_date.isoformat() if e.entry_date else None,
            'mood_score': e.mood_score,
            'emotion': emotion_label,
            'secondary_emotion': secondary_emotion,
            'triggered_keywords': triggered,
            'created_at': e.created_at.isoformat() if e.created_at else None
        })
    return jsonify({'entries': result})

## Removed /journal/new route in favor of modal-based journaling on dashboard

@app.route('/api/patterns', methods=['GET'])
def api_patterns():
    if 'user' not in session:
        return jsonify({'error': 'authentication required'}), 401
    user = User.query.filter_by(name=session['user']).first()
    if not user:
        return jsonify({'error': 'user not found'}), 404

    entries = JournalEntry.query.filter_by(user_id=user.id).order_by(JournalEntry.created_at.asc()).all()
    if not entries:
        return jsonify({'message': 'no data', 'triggers_high': [], 'triggers_low': [], 'weekday_avg': {}, 'worst_day': None, 'trend': 'stable', 'slope': 0.0, 'n_entries': 0})

    scores = []
    dates = []
    word_sum = {}
    word_cnt = {}
    stop = {
        'the','a','an','and','or','but','if','in','on','at','to','for','with','of','is','it','this','that','was','were','be','am','are','i','you','he','she','they','we','me','my','your','our','their','from','as','by','so','not','no','do','did','does','have','had','has'
    }

    for e in entries:
        s = e.mood_score if e.mood_score is not None else 0.0
        scores.append(float(s))
        dates.append(e.created_at or datetime.now())
        text = (e.entry_text or '').lower()
        words = re.findall(r"[a-z']+", text)
        seen = set()
        for w in words:
            if len(w) < 3 or w in stop:
                continue
            # count once per entry to avoid long texts dominating
            if w in seen:
                continue
            seen.add(w)
            word_sum[w] = word_sum.get(w, 0.0) + s
            word_cnt[w] = word_cnt.get(w, 0) + 1

    min_count = 3
    word_avg = [(w, word_sum[w] / word_cnt[w]) for w in word_cnt if word_cnt[w] >= min_count]
    word_avg.sort(key=lambda x: x[1])
    triggers_low = [(w, round(avg, 3)) for w, avg in word_avg[:10]]
    word_avg.sort(key=lambda x: x[1], reverse=True)
    triggers_high = [(w, round(avg, 3)) for w, avg in word_avg[:10]]

    weekday_groups = {i: [] for i in range(7)}
    for s, d in zip(scores, dates):
        weekday_groups[d.weekday()].append(s)
    weekday_avg = {str(i): (round(float(np.mean(v)), 4) if v else None) for i, v in weekday_groups.items()}
    worst_day = None
    filtered = [(i, np.mean(v)) for i, v in weekday_groups.items() if v]
    if filtered:
        worst_day = int(min(filtered, key=lambda x: x[1])[0])

    slope = 0.0
    trend = 'stable'
    if len(scores) >= 2:
        x = np.arange(len(scores))
        slope = float(np.polyfit(x, np.array(scores, dtype=float), 1)[0])
        if slope > 0.01:
            trend = 'improving'
        elif slope < -0.01:
            trend = 'declining'

    return jsonify({
        'triggers_high': triggers_high,
        'triggers_low': triggers_low,
        'weekday_avg': weekday_avg,
        'worst_day': worst_day,
        'trend': trend,
        'slope': round(slope, 5),
        'n_entries': len(entries)
    })

def generate_personalized_insights(user_id):
    """Generate AI insights from user's mood data"""
    user = User.query.filter_by(id=user_id).first()
    if not user:
        return []

    entries = JournalEntry.query.filter_by(user_id=user_id).order_by(JournalEntry.created_at.desc()).limit(30).all()
    if len(entries) < 5:
        return []

    insights = []

    # INSIGHT 1: Best/Worst days of week
    weekday_moods = {i: [] for i in range(7)}
    for e in entries:
        if e.mood_score is not None and e.created_at:
            weekday_moods[e.created_at.weekday()].append(float(e.mood_score))

    weekday_avg = {i: np.mean(v) for i, v in weekday_moods.items() if v}
    if weekday_avg:
        best_day = max(weekday_avg, key=weekday_avg.get)
        worst_day = min(weekday_avg, key=weekday_avg.get)
        best_name = calendar.day_name[best_day]
        worst_name = calendar.day_name[worst_day]

        insights.append({
            'type': 'best_day',
            'text': f"🏆 {best_name}s are your best days! Average mood: {weekday_avg[best_day]:.2f}",
            'value': float(weekday_avg[best_day])
        })
        insights.append({
            'type': 'worst_day',
            'text': f"⚠️ {worst_name}s need extra care. Average mood: {weekday_avg[worst_day]:.2f}",
            'value': float(weekday_avg[worst_day])
        })

    # INSIGHT 2: Top positive/negative triggers (context-aware)
    def _extract_triggers(text):
        if not text:
            return []
        t = text.lower()
        found = []
        triggers = {
            'exercise': ['exercise','gym','ran','walked','workout','sport','fitness'],
            'social': ['friend','friends','family','hang out','party','social','met','visited'],
            'accomplishment': ['finished','completed','achieved','succeeded','won','got','promotion','project'],
            'relaxation': ['relax','rest','sleep','vacation','break','chill','calm','peaceful'],
            'nature': ['nature','park','outside','weather','sun','outdoor','hiking'],
            'stress': ['stress','stressed','pressure','busy','rushed','overwhelmed','deadline'],
            'conflict': ['argument','fight','angry','upset','mad','frustrated','conflict'],
            'fatigue': ['tired','exhausted','sleep','sleepy','fatigue','worn out'],
            'health': ['sick','ill','pain','hurt','cold','flu','headache','unwell'],
            'loneliness': ['alone','lonely','isolated','nobody','no one','abandoned'],
            'failure': ['failed','fail','mistake','wrong','loss','lost',"didn't work"],
        }
        for name, kws in triggers.items():
            if any(kw in t for kw in kws):
                found.append(name)
        return found

    trigger_counts = {'positive': {}, 'negative': {}}
    for e in entries:
        if e.entry_text:
            tgs = _extract_triggers(e.entry_text)
            mood_val = float(e.mood_score) if e.mood_score is not None else 0.5
            for g in tgs:
                if mood_val > 0.6:
                    trigger_counts['positive'][g] = trigger_counts['positive'].get(g, 0) + 1
                elif mood_val < 0.4:
                    trigger_counts['negative'][g] = trigger_counts['negative'].get(g, 0) + 1

    if trigger_counts['positive']:
        top_positive = max(trigger_counts['positive'], key=trigger_counts['positive'].get)
        count = trigger_counts['positive'][top_positive]
        insights.append({
            'type': 'positive_trigger',
            'text': f"✨ {top_positive.capitalize()} appeared in {count} of your happy entries!",
            'value': int(count)
        })

    if trigger_counts['negative']:
        top_negative = max(trigger_counts['negative'], key=trigger_counts['negative'].get)
        count = trigger_counts['negative'][top_negative]
        insights.append({
            'type': 'negative_trigger',
            'text': f"⚡ {top_negative.capitalize()} appears when you're down. Consider managing it.",
            'value': int(count)
        })

    # INSIGHT 3: Consistency bonus
    mood_entries = [e for e in entries if e.mood_score is not None]
    consistency_rate = (len(mood_entries) / len(entries)) * 100 if entries else 0
    if consistency_rate > 80:
        insights.append({
            'type': 'consistency',
            'text': f"🔥 Amazing consistency! You've logged {len(mood_entries)} moods in {len(entries)} days!",
            'value': float(consistency_rate)
        })

    return insights[:5]

@app.route('/api/insights/personalized', methods=['GET'])
def api_personalized_insights():
    if 'user' not in session:
        return jsonify({'error': 'authentication required'}), 401

    user = User.query.filter_by(name=session['user']).first()
    if not user:
        return jsonify({'error': 'user not found'}), 404

    insights = generate_personalized_insights(user.id)
    return jsonify({
        'status': 'success',
        'insights': insights
    })

@app.route('/api/predict', methods=['GET'])
def api_predict():
    if 'user' not in session:
        return jsonify({'error': 'authentication required'}), 401
    user = User.query.filter_by(name=session['user']).first()
    if not user:
        return jsonify({'error': 'user not found'}), 404

    # Last 30 entries (ascending for time calculations)
    last_30 = JournalEntry.query.filter_by(user_id=user.id).order_by(JournalEntry.created_at.desc()).limit(30).all()
    last_30 = list(reversed(last_30))
    scores = [float(e.mood_score) for e in last_30 if e.mood_score is not None]
    if len(scores) < 3:
        return jsonify({'status': 'insufficient_data', 'message': 'Need at least 3 mood entries to predict'}), 200

    # Baseline from last 7 entries within last 30
    recent_for_baseline = scores[-7:] if len(scores) >= 7 else scores
    baseline = float(np.mean(recent_for_baseline)) if recent_for_baseline else 0.0

    # Weekday averages from last 30
    weekday_groups = {i: [] for i in range(7)}
    for e in last_30:
        if e.mood_score is None:
            continue
        d = e.created_at or datetime.now()
        weekday_groups[d.weekday()].append(float(e.mood_score))
    weekday_avg = {i: (float(np.mean(v)) if v else None) for i, v in weekday_groups.items()}

    # Trend via slope
    slope = 0.0
    if len(scores) >= 2:
        x = np.arange(len(scores))
        slope = float(np.polyfit(x, np.array(scores, dtype=float), 1)[0])
    trend_direction = 'stable'
    if slope > 0.01:
        trend_direction = 'improving'
    elif slope < -0.01:
        trend_direction = 'declining'

    # Context-aware trigger detection with weights
    def extract_triggers(text):
        """Extract top 3 context-aware triggers by absolute weight."""
        if not text:
            return []

        text_lower = text.lower()
        found_triggers = {}

        triggers = {
            # POSITIVE TRIGGERS
            'exercise': {'keywords': ['exercise', 'gym', 'ran', 'walked', 'workout', 'sport', 'fitness'], 'weight': 0.3},
            'social': {'keywords': ['friend', 'friends', 'family', 'hang out', 'party', 'social', 'met', 'visited'], 'weight': 0.3},
            'accomplishment': {'keywords': ['finished', 'completed', 'achieved', 'succeeded', 'won', 'got', 'promotion', 'project'], 'weight': 0.25},
            'relaxation': {'keywords': ['relax', 'rest', 'sleep', 'vacation', 'break', 'chill', 'calm', 'peaceful'], 'weight': 0.2},
            'nature': {'keywords': ['nature', 'park', 'outside', 'weather', 'sun', 'outdoor', 'hiking'], 'weight': 0.15},

            # NEGATIVE TRIGGERS
            'stress': {'keywords': ['stress', 'stressed', 'pressure', 'busy', 'rushed', 'overwhelmed', 'deadline'], 'weight': -0.3},
            'conflict': {'keywords': ['argument', 'fight', 'angry', 'upset', 'mad', 'frustrated', 'conflict'], 'weight': -0.35},
            'fatigue': {'keywords': ['tired', 'exhausted', 'sleep', 'sleepy', 'fatigue', 'worn out'], 'weight': -0.2},
            'health': {'keywords': ['sick', 'ill', 'pain', 'hurt', 'cold', 'flu', 'headache', 'unwell'], 'weight': -0.35},
            'loneliness': {'keywords': ['alone', 'lonely', 'isolated', 'nobody', 'no one', 'abandoned'], 'weight': -0.4},
            'failure': {'keywords': ['failed', 'fail', 'mistake', 'wrong', 'loss', 'lost', "didn't work"], 'weight': -0.3},
        }

        for trigger_name, trigger_data in triggers.items():
            for keyword in trigger_data['keywords']:
                if keyword in text_lower:
                    found_triggers[trigger_name] = trigger_data['weight']
                    break

        sorted_triggers = sorted(found_triggers.items(), key=lambda x: abs(x[1]), reverse=True)
        return [t[0] for t in sorted_triggers[:3]]

    def trigger_weight(name: str) -> float:
        weights = {
            'exercise': 0.3, 'social': 0.3, 'accomplishment': 0.25, 'relaxation': 0.2, 'nature': 0.15,
            'stress': -0.3, 'conflict': -0.35, 'fatigue': -0.2, 'health': -0.35, 'loneliness': -0.4, 'failure': -0.3,
        }
        return float(weights.get(name, 0.0))

    recent5 = last_30[-5:]
    all_recent_triggers = []
    total_adjust = 0.0
    for e in recent5:
        trig_names = extract_triggers(e.entry_text)
        all_recent_triggers.extend(trig_names)
        adj = sum(trigger_weight(t) for t in trig_names)
        total_adjust += adj
    trigger_count_den = len(recent5) if recent5 else 0
    trigger_score_base = (total_adjust / trigger_count_den) if trigger_count_den > 0 else 0.0

    # Build predictions for next 7 days
    predictions = []
    today = date.today()
    for i in range(1, 8):
        target_date = today + timedelta(days=i)
        dow = target_date.weekday()
        day_name = target_date.strftime('%A')
        day_avg = weekday_avg.get(dow)
        day_adjustment = 0.0
        if day_avg is not None:
            day_adjustment = (day_avg - baseline) * 0.4

        trend_adjustment = slope * i
        # BETTER FORMULA: Add momentum and weighted recency
        recent_scores = scores[-3:] if len(scores) >= 3 else scores
        recent_average = float(np.mean(recent_scores)) if recent_scores else baseline
        # Momentum over last few entries
        if len(recent_scores) >= 3:
            momentum = float(recent_scores[-1]) - float(recent_scores[0])
        elif len(recent_scores) >= 2:
            momentum = float(recent_scores[-1]) - float(recent_scores[-2])
        else:
            momentum = 0.0
        # Blend baseline with recent trend (30% recent, 70% baseline)
        weighted_baseline = (baseline * 0.7) + (recent_average * 0.3)
        # Consistency bonus: regular logging stabilizes predictions
        entry_count = len(scores)
        if entry_count >= 15:
            stability_bonus = 0.05
        elif entry_count >= 8:
            stability_bonus = 0.02
        else:
            stability_bonus = 0.0
        # Build improved raw prediction
        predicted_raw = weighted_baseline + day_adjustment + (trigger_score_base * 0.8) + trend_adjustment + (momentum * 0.2) + stability_bonus
        # Map from [-1, 1] to [0, 1] so negatives don't collapse to 0
        predicted = (predicted_raw + 1.0) / 2.0
        # Clamp to valid range
        predicted = max(0.0, min(1.0, float(predicted)))

        # SMARTER CONFIDENCE: Based on data quality
        base_confidence = 0.80
        # Reduce confidence based on distance (further in future = less certain)
        distance_penalty = i * 0.04
        base_confidence -= distance_penalty
        # BOOST confidence if we have good data for this weekday
        dow_entries = weekday_groups.get(dow, [])
        if len(dow_entries) >= 7:
            base_confidence += 0.10  # Strong pattern
        elif len(dow_entries) >= 4:
            base_confidence += 0.05  # Moderate pattern
        elif len(dow_entries) <= 1:
            base_confidence -= 0.15  # Weak pattern
        # Overall data volume bonus
        if len(scores) >= 20:
            base_confidence += 0.08  # Rich dataset
        elif len(scores) >= 10:
            base_confidence += 0.04  # Decent dataset
        elif len(scores) < 5:
            base_confidence -= 0.10  # Sparse data
        confidence = max(0.3, min(0.95, base_confidence))  # Keep in reasonable range

        # Reasoning per-day
        triggers_found = sorted(set(all_recent_triggers))
        parts = []
        if day_avg is not None:
            parts.append(f"{day_name} is typically {'positive' if day_avg >= baseline else 'lower'} ({day_avg:.2f}).")
        if triggers_found:
            parts.append("Recent journals mention triggers: " + ", ".join(triggers_found) + ".")
        parts.append(f"Your mood trend is {trend_direction}.")
        parts.append(f"Prediction: {predicted:.2f}")
        reasoning = " ".join(parts)

        predictions.append({
            'date': target_date.isoformat(),
            'day_name': day_name,
            'predicted_mood': round(predicted, 3),
            'confidence': round(confidence, 3),
            'day_avg': (round(day_avg, 3) if day_avg is not None else None),
            'triggers_found': triggers_found,
            'reasoning': reasoning
        })

    overall_reasoning = (
        f"Your baseline is {baseline:.2f}. Trend is {trend_direction} ({slope:+.2f}/day). "
    )
    # Worst weekday over last 30
    worst_day = None
    filtered = [(i, np.mean(v)) for i, v in weekday_groups.items() if v]
    if filtered:
        worst_idx, worst_val = min(filtered, key=lambda x: x[1])
        worst_day = (worst_idx, float(worst_val))
        overall_reasoning += f"{calendar.day_name[worst_idx]} is your hardest day ({worst_val:.2f}). "

    if all_recent_triggers:
        overall_reasoning += "Triggers affecting you recently: " + ", ".join(sorted(set(all_recent_triggers))) + "."

    recommendation = "Your mood is stable. Keep consistent habits."
    if trend_direction == 'declining':
        recommendation = "Try exercise or social activity, and plan breaks to manage stress."
    elif trend_direction == 'improving':
        recommendation = "Your mood is improving! Keep exercising and spending time with friends."
    if worst_day is not None and (worst_day[1] < baseline):
        recommendation += f" Be prepared for {calendar.day_name[worst_day[0]]} dips."

    # Debug logging for prediction internals (first 2 days example)
    try:
        print("\n=== PREDICTION DEBUG ===")
        print(f"Baseline: {baseline}")
        print(f"Trend slope: {slope}")
        print(f"Trigger score base: {trigger_score_base}")
        print(f"Weekday averages: {weekday_avg}")
        print("\nFirst prediction calculation:")
        for i in range(1, 3):
            target_date_dbg = today + timedelta(days=i)
            dow_dbg = target_date_dbg.weekday()
            day_avg_dbg = weekday_avg.get(dow_dbg)
            day_adj_dbg = (day_avg_dbg - baseline) * 0.4 if day_avg_dbg is not None else 0.0
            trend_adj_dbg = slope * i
            predicted_dbg = baseline + day_adj_dbg + trigger_score_base + trend_adj_dbg
            print(f"Day {i}: baseline({baseline}) + day_adj({day_adj_dbg}) + trigger({trigger_score_base}) + trend({trend_adj_dbg}) = {predicted_dbg}")
        print("=== END DEBUG ===\n")
    except Exception as _dbg_err:
        try:
            print('[predict][debug] logging failed:', _dbg_err)
        except Exception:
            pass

    return jsonify({
        'status': 'success',
        'baseline': round(baseline, 3),
        'trend': trend_direction,
        'predictions': predictions,
        'overall_reasoning': overall_reasoning.strip(),
        'recommendation': recommendation
    })

@app.route('/predictions')
def predictions_dashboard():
    if 'user' not in session:
        return redirect('/login')
    return render_template('dashboard_predictions.html', user=session.get('user'))

@app.route('/api/onboarding/status', methods=['GET'])
def onboarding_status():
    seen = bool(session.get('onboarding_seen', False))
    return jsonify({'seen': seen})

@app.route('/api/onboarding/seen', methods=['POST'])
def onboarding_seen():
    session['onboarding_seen'] = True
    return jsonify({'status': 'ok'})

# Run app
if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)
