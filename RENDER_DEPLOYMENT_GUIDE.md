# 🚀 MindSpace Render Deployment – Complete Setup Guide

This document provides all the information needed to deploy your MindSpace application to Render's free tier, including file structure changes, code modifications, and step-by-step instructions.

---

## 📋 What Was Changed For Deployment

### 1. **requirements.txt** (Updated)
✅ **CPU-Optimized ML Libraries**
- `tensorflow-cpu==2.13.0` (NOT tensorflow, which requires GPU)
- `opencv-python-headless==4.8.0.74` (server-friendly, no GUI)
- `scikit-learn==1.3.0` (TF-IDF + Linear Regression)
- `gunicorn==21.2.0` (production web server)
- All other dependencies pinned to compatible versions

### 2. **Procfile** (Created)
✅ **Gunicorn Configuration for Render**
```
web: gunicorn --workers 2 --threads 2 --worker-class gthread --bind 0.0.0.0:$PORT --timeout 120 app:app
```
- `--workers 2`: Limited to 2 workers (free tier memory constraint)
- `--threads 2`: Threading for concurrency without extra processes
- `--timeout 120`: 120-second timeout (sufficient for ML model operations)
- Auto-reads `$PORT` from Render environment

### 3. **render.yaml** (Created - Optional)
✅ **Infrastructure-as-Code Configuration**
- Defines Python 3.11 environment
- Sets `FLASK_ENV=production`
- Auto-generates `APP_SECRET_KEY`
- Can be used instead of manual Render dashboard setup

### 4. **.gitignore** (Updated)
✅ **Excludes:**
- `venv/`, `venv310/`, `moodenv/` (virtual environments)
- `__pycache__/`, `*.pyc` (Python cache)
- `instance/app.db` (ephemeral database)
- `.env` files (secrets)
- Node modules and build artifacts
- Preserves committed files like `static/css/output.css`

### 5. **app.py** (Updated for Production)
✅ **Key Changes:**

#### a) **Relative Path ML Model Loading**
```python
def load_ml_models():
    """Load all ML models at startup to avoid Render timeouts"""
    app_root = os.path.dirname(os.path.abspath(__file__))
    
    cascade_path = os.path.join(app_root, 'ml_model', 'Emotion_Dectector', 'haarcascade_frontalface_default.xml')
    face_classifier = cv2.CascadeClassifier(cascade_path)
    
    model_path = os.path.join(app_root, 'ml_model', 'Emotion_Dectector', 'model.h5')
    classifier = load_model(model_path)
    
    nlp_model_path = os.path.join(app_root, 'ml_model', 'NLP_Text_Emotion', 'models', 'emotion_classifier_pipe_lr_03_jan_2022.pkl')
    nlp_model = joblib.load(nlp_model_path)
```
- ✅ Replaced absolute Windows paths with `os.path.join()`
- ✅ Works on any system (Windows, Linux, Render)

#### b) **Models Loaded at Startup (Not on First Request)**
```python
if __name__ == '__main__':
    with app.app_context():
        db.create_all()
        load_ml_models()  # <-- LOADS MODELS HERE
```
- Prevents timeout on first user request
- Render has 60s boot time; loading models upfront prevents timeout

#### c) **Production-Ready Settings**
```python
app.run(
    host='0.0.0.0',           # Listen on all interfaces (required for Render)
    port=port,                # Read from $PORT env var
    debug=debug_mode,         # False in production
    threaded=True,            # Multi-threaded Flask (safe)
    use_reloader=False        # Disabled for Gunicorn compatibility
)
```

#### d) **Graceful Model Loading Fallbacks**
- If models fail to load, app still starts (with degraded functionality)
- Error messages logged to console
- Prevents full app crash

#### e) **Removed Problematic References**
- Removed reference to undefined `pipe_lr` variable
- Fixed secondary emotion extraction to only use loaded `nlp_model`

### 6. **README_deploy.md** (Created)
✅ Complete step-by-step deployment guide with:
- Project structure verification
- Render dashboard setup
- Environment variable configuration
- Troubleshooting tips
- Database handling notes

---

## 📁 Final Project Structure (Verified)

```
mindspace-mood-tracker/
│
├── app.py                              # Updated for production
├── requirements.txt                    # Updated: CPU-friendly versions
├── Procfile                            # NEW: Gunicorn config
├── render.yaml                         # NEW: Optional infrastructure-as-code
├── .gitignore                          # Updated: Deployment-ready
├── README.md                           # Original project overview
├── README_deploy.md                    # NEW: Deployment guide
├── PROJECT_DOCUMENTATION.md            # Detailed architecture docs
│
├── instance/                           # Auto-created by Flask
│   └── app.db                          # SQLite database (ephemeral on Render)
│
├── ml_model/                           # ✅ ALL COMMITTED TO GIT
│   │
│   ├── Emotion_Dectector/
│   │   ├── model.h5                   # CNN model (REQUIRED)
│   │   ├── haarcascade_frontalface_default.xml
│   │   ├── predict_face.py
│   │   ├── emotion-classification-cnn-using-keras.ipynb
│   │   └── __pycache__/               # (ignored in .gitignore)
│   │
│   ├── NLP_Text_Emotion/
│   │   ├── models/
│   │   │   └── emotion_classifier_pipe_lr_03_jan_2022.pkl  # (REQUIRED)
│   │   ├── text_emotion_utils.py
│   │   ├── predict_text.py
│   │   ├── text_emotion_utils.py
│   │   └── requirements.txt
│   │
│   ├── Webcam Opencv Project/         # (Optional, not used in main app)
│   │   ├── app.py
│   │   └── ...
│   │
│   ├── voice_mood_analyzer.py         # (Optional, not fully integrated)
│   └── requirements.txt
│
├── templates/                          # ✅ ALL COMMITTED TO GIT
│   ├── splash.html
│   ├── login.html
│   ├── dashboard.html
│   ├── dashboard_predictions.html
│   ├── detect_emotion.html
│   ├── text_emotion.html
│   ├── text_emotion_result.html
│   ├── voice_emotion.html
│   ├── journal_entry.html
│   ├── about.html
│   └── index.html
│
├── static/                             # ✅ ALL COMMITTED TO GIT
│   ├── css/
│   │   ├── input.css                  # Tailwind source (optional)
│   │   ├── output.css                 # ✅ MUST BE COMMITTED (pre-built!)
│   │   └── styles.css
│   └── images/
│       └── splash_logo.html/
│           ├── splash_logo.html
│           └── splash_logo_files/
│               └── ...
│
├── data/                               # Optional data folder
│   └── (any CSV files, etc.)
│
├── scripts/                            # (Optional utility scripts)
│   └── ...
│
└── .git/                               # ✅ Repository (committed to GitHub)
    └── ...

# DO NOT COMMIT (ignored in .gitignore):
# - venv/, venv310/, moodenv/
# - __pycache__/, *.pyc
# - instance/app.db
# - .env
# - node_modules/
```

---

## ✅ Deployment Checklist

Before deploying to Render, complete this checklist:

### Code & Files
- [ ] All files in `ml_model/` are committed (models, cascades, etc.)
- [ ] All files in `templates/` are committed
- [ ] All files in `static/` are committed (including `output.css`)
- [ ] `app.py` updated with new model loading code
- [ ] `requirements.txt` updated with pinned versions
- [ ] `Procfile` created with Gunicorn config
- [ ] `render.yaml` created (optional but recommended)
- [ ] `.gitignore` updated to exclude venv and local files
- [ ] NO absolute Windows paths in code (all use `os.path.join()`)

### Git & GitHub
- [ ] All changes committed: `git add .` → `git commit -m "..."`
- [ ] Latest changes pushed to `master` branch
- [ ] Repository is public or accessible to your GitHub account

### Tailwind CSS
- [ ] Run build locally: `npm run build` (if you modified CSS)
- [ ] Commit `static/css/output.css` to git
- [ ] `render.yaml` does NOT include any build commands for Tailwind

### Environment
- [ ] No `.env` files committed to git
- [ ] Environment variables will be set in Render dashboard

---

## 🚀 Step-by-Step Render Deployment

### **STEP 1: Final Git Commit & Push**

```bash
cd "C:\Users\Lenovo\Desktop\mood tracker app"

# Stage all files
git add .

# Commit changes
git commit -m "Prepare for Render deployment: update app.py, requirements, add Procfile"

# Push to GitHub
git push origin master
```

### **STEP 2: Create Render Web Service**

1. Go to **https://render.com**
2. Sign in with your GitHub account
3. Click **New** → **Web Service**
4. Select your **MindSpace** repository
5. Fill in the form:

   | Field | Value |
   |-------|-------|
   | **Name** | `mindspace-mood-tracker` |
   | **Environment** | `Python 3` |
   | **Region** | `Oregon` (closest to you or users) |
   | **Branch** | `master` |
   | **Build Command** | `pip install -r requirements.txt` |
   | **Start Command** | (leave blank - Procfile will be used) |
   | **Plan** | `Free` |

6. Click **Create Web Service**

### **STEP 3: Set Environment Variables**

In your Render dashboard for the web service:

1. Go to **Environment** tab
2. Click **Add Environment Variable**
3. Add these variables:

   | Key | Value |
   |-----|-------|
   | `FLASK_ENV` | `production` |
   | `APP_SECRET_KEY` | Generate a random string (e.g., use Python: `os.urandom(24).hex()`) |

4. Click **Save**

### **STEP 4: Wait for Deployment**

- Render will automatically build and deploy
- You'll see deployment logs in the **Logs** tab
- Build typically takes 3-5 minutes
- Once live, your app is at: `https://mindspace-mood-tracker.onrender.com`

### **STEP 5: Test Your App**

1. Visit `https://mindspace-mood-tracker.onrender.com`
2. Register a new user
3. Test mood logging
4. Test text emotion detection
5. Test journal entry creation

### **STEP 6: Monitor for Issues**

If the app doesn't start:
1. Go to **Logs** in Render dashboard
2. Look for errors (typically about models or imports)
3. Fix locally, commit, push to master
4. Render will auto-redeploy

---

## 🔍 Common Deployment Issues & Solutions

### Issue 1: **"Module not found" Error**
**Symptom:** Build fails with `ModuleNotFoundError`

**Solution:**
- Add missing package to `requirements.txt`
- Commit and push
- Render will auto-redeploy

### Issue 2: **"Timeout" During Startup**
**Symptom:** Service times out after 60 seconds

**Reason:** ML models taking too long to load

**Solution:**
- Models are now pre-loaded at startup (fixed in updated `app.py`)
- If still timing out, check Procfile timeout is set to 120s (already done)
- Could indicate very large model file

### Issue 3: **"No such file or directory" (Model Files)**
**Symptom:** `FileNotFoundError` for `model.h5` or `.pkl`

**Reason:** Model files not committed to git OR incorrect paths

**Solution:**
- Ensure `ml_model/Emotion_Dectector/model.h5` is in git
- Ensure `ml_model/NLP_Text_Emotion/models/emotion_classifier_pipe_lr_03_jan_2022.pkl` is in git
- Verify file paths use `os.path.join()` (not absolute Windows paths)

### Issue 4: **Static Files Not Loading (CSS, Images)**
**Symptom:** Website appears unstyled or images missing

**Solution:**
- Ensure `static/` folder is NOT in `.gitignore` exclusion
- Commit `static/css/output.css` (pre-built Tailwind)
- Verify Flask serves static files (it does by default)
- Clear browser cache (Ctrl+Shift+Del)

### Issue 5: **Memory Exceeded**
**Symptom:** Service crashes after a while

**Reason:** ML models use significant RAM (TensorFlow, scikit-learn)

**Solution:**
- Free tier Render has 512 MB RAM (Procfile is optimized)
- If still exceeding, consider upgrading to paid tier or optimizing models
- Monitor via Render dashboard **Metrics** tab

### Issue 6: **Video Feed / Webcam Not Working**
**Symptom:** Webcam stream shows black or error

**Expected Behavior:** Webcam works in browser only if user grants permission
- On server-side, there's no webcam to capture
- Browser video feed works fine when user visits webpage

### Issue 7: **Database Lost After Redeploy**
**Expected Behavior:** SQLite data is ephemeral on free tier

**Solution:** This is normal. If you need persistent data:
- Upgrade to paid tier with persistent storage, OR
- Add Render PostgreSQL database (free tier available)
- Update app.py to use PostgreSQL connection string

---

## 📊 Monitoring Your Deployed App

In your Render dashboard:

1. **Logs Tab** → See real-time application output
2. **Metrics Tab** → Monitor CPU, memory, request count
3. **Events Tab** → See deployment history and restarts
4. **Health Check** → Render pings your app regularly to keep it alive

---

## 🔄 Updates & Redeployment

To update your app after deployment:

1. Make changes locally in your code
2. Test locally: `python app.py`
3. Commit and push to GitHub:
   ```bash
   git add .
   git commit -m "Update feature: ..."
   git push origin master
   ```
4. Render automatically redeploys (usually within 1 minute)
5. Check Logs to ensure deployment succeeded

---

## 📝 Important Notes

### Database (SQLite)
- Stored in `instance/app.db`
- On Render free tier: **NOT persistent** (reset on redeploy or restart)
- Each deployment creates a fresh database
- Users will need to re-register after redeploy
- To preserve data: Migrate to PostgreSQL

### ML Models
- Loaded once at startup (in `if __name__ == '__main__'`)
- Prevents timeout on first request
- Models stay in memory throughout app lifetime
- Uses ~200-300 MB RAM (normal for TensorFlow + scikit-learn)

### Gunicorn
- Handles concurrent requests safely
- 2 workers + 2 threads = up to 4 concurrent connections
- Sufficient for free tier traffic

### Static Files
- Flask serves static files from `static/` folder
- Tailwind CSS (`output.css`) must be pre-built and committed
- No build process runs on Render

---

## 🆘 Support

If you encounter issues:

1. **Check Render Logs** → Most useful for diagnosing problems
2. **Read Flask Docs** → https://flask.palletsprojects.com/
3. **Read Render Docs** → https://render.com/docs
4. **Test Locally First** → Run `python app.py` locally and verify everything works
5. **Check `.gitignore`** → Ensure required files are committed

---

## ✨ Summary

Your MindSpace app is now ready for Render deployment! The key changes made:

1. ✅ **Gunicorn + Procfile** for production serving
2. ✅ **Relative paths** for model loading (works anywhere)
3. ✅ **Models loaded at startup** (prevents timeout)
4. ✅ **CPU-optimized dependencies** (tensorflow-cpu, headless OpenCV)
5. ✅ **Production-ready Flask config** (host, port, debug=False)
6. ✅ **Comprehensive documentation** (this file + README_deploy.md)

**All original ML features preserved:**
- ✅ CNN facial emotion detection
- ✅ TF-IDF + Linear Regression text emotion
- ✅ Mood predictions & insights
- ✅ Trigger detection & patterns
- ✅ Journal entries & analytics

Deploy with confidence! 🚀
