# MindSpace - Render Deployment Guide (Free Tier)

This guide walks you through deploying MindSpace to Render's free tier without any paid resources or external services.

## ✅ Prerequisites

- GitHub account with the MindSpace repository
- Render account (free tier)
- Project structure following the organization below

## 📁 Project Structure (Verified)

```
mood-tracker-app/
├── app.py                      # Main Flask app (updated for Gunicorn)
├── requirements.txt            # Dependencies (Render-optimized)
├── Procfile                    # Gunicorn configuration
├── render.yaml                 # Render infrastructure definition
├── .gitignore                  # Git ignore rules
├── README.md                   # Project overview
├── README_deploy.md            # This file
│
├── instance/                   # Auto-created by Flask
│   └── app.db                  # SQLite database (ephemeral on free tier)
│
├── ml_model/                   # ML models
│   ├── Emotion_Dectector/
│   │   ├── model.h5           # Face CNN model (committed to git)
│   │   ├── haarcascade_frontalface_default.xml
│   │   └── predict_face.py
│   │
│   ├── NLP_Text_Emotion/
│   │   ├── models/
│   │   │   └── emotion_classifier_pipe_lr_03_jan_2022.pkl
│   │   ├── text_emotion_utils.py
│   │   ├── predict_text.py
│   │   └── requirements.txt
│   │
│   ├── voice_mood_analyzer.py
│   └── requirements.txt
│
├── templates/                  # HTML templates
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
├── static/                     # Static assets (pre-built!)
│   ├── css/
│   │   ├── input.css          # Source (optional)
│   │   ├── output.css         # MUST BE COMMITTED (no build on Render)
│   │   └── styles.css
│   └── images/
│       └── splash_logo.html/
│
├── scripts/                    # Optional utility scripts
│
└── data/                       # Optional data folder
```

## 🚀 Step-by-Step Deployment

### Step 1: Prepare Your Repository

1. **Ensure all files are committed to GitHub:**
   ```bash
   git add .
   git commit -m "Prepare for Render deployment"
   git push origin master
   ```

2. **Verify Tailwind CSS is pre-built:**
   - The `static/css/output.css` file must be committed
   - Do NOT rely on Render to build Tailwind
   - If you modified styles, rebuild locally:
     ```bash
     npm run build  # or your Tailwind build command
     git add static/css/output.css
     git commit -m "Update Tailwind CSS"
     git push
     ```

3. **Verify `.gitignore` includes:**
   - `venv/`, `__pycache__/`, `*.pyc`
   - `instance/app.db` (database will be ephemeral)
   - `node_modules/`

### Step 2: Create a Render Web Service

1. Go to [Render.com](https://render.com)
2. Sign in with your GitHub account
3. Click **New** → **Web Service**
4. Select your MindSpace repository
5. Configure as follows:

   | Setting | Value |
   |---------|-------|
   | **Name** | `mindspace-mood-tracker` (or your choice) |
   | **Environment** | `Python 3` |
   | **Build Command** | `pip install -r requirements.txt` |
   | **Start Command** | Leave blank (Procfile will be used) |
   | **Plan** | Free |

6. Click **Create Web Service**

### Step 3: Set Environment Variables

In your Render dashboard, go to **Environment** and add:

| Key | Value |
|-----|-------|
| `FLASK_ENV` | `production` |
| `APP_SECRET_KEY` | `your-secret-key-here` (generate a random string) |

**Note:** Render auto-generates `PORT` (no need to set)

### Step 4: Deploy

1. Render will automatically build and deploy from your `master` branch
2. Wait 2-3 minutes for the build to complete
3. Once deployed, visit your app at `https://mindspace-mood-tracker.onrender.com` (or your custom domain)

### Step 5: Monitor Logs

- Go to **Logs** in your Render dashboard
- Check for any errors during startup
- Common issues:
  - **Model loading timeout**: Increase `timeout` in Procfile (already set to 120s)
  - **Out of memory**: Use `--workers 2` (already configured for free tier)
  - **Static files not found**: Ensure `static/` folder is committed

## 🔄 Deployment Options

### Option A: Auto-Deploy on Push
- Render will automatically redeploy whenever you push to `master`
- Best for active development

### Option B: Manual Deploy
- In Render dashboard, click **Manual Deploy** → **Deploy latest commit**
- Use this for testing before auto-deploying

## 💾 Database Notes

**Important:** Render's free tier does NOT include persistent storage for SQLite.

### What happens to data?
- Each deployment creates a fresh `instance/app.db`
- Data persists during the service lifetime (until restart/redeploy)
- Data is lost when:
  - Service is paused
  - Service is redeployed
  - Render restarts the dyno

### Recommended: Add PostgreSQL (Optional, Free)
If you want persistent data:
1. Create a Render PostgreSQL database (free tier available)
2. Update `DATABASE_URL` in environment
3. Modify `app.py` to use PostgreSQL connection string
4. Run migrations

For now, SQLite is sufficient for demo purposes.

## 🔧 Troubleshooting

### Build Fails: "Module not found"
- **Cause:** Missing dependency in `requirements.txt`
- **Fix:** Add to `requirements.txt`, commit, push, redeploy

### App crashes after deploy
- **Cause:** ML models too large or taking too long to load
- **Fix:** Check Procfile `timeout` (set to 120s). If still failing, optimize model loading
- **View logs:** Go to Render dashboard → Logs

### Static files (CSS, images) not loading
- **Cause:** `static/css/output.css` not committed OR Flask not serving static files
- **Fix:** 
  - Ensure `static/` is in `.gitignore` EXCLUSION (not ignored)
  - Commit all CSS files
  - Verify Flask is configured to serve from `static/` folder

### Video feed / webcam not working
- **Cause:** OpenCV cannot access webcam on server
- **Expected:** On server, webcam access is limited. Users must allow webcam in browser
- **Note:** This is normal. Video feed only works for browser users, not server-side

### High memory usage
- **Cause:** ML models (TensorFlow, scikit-learn) loaded in memory
- **Expected:** This is normal for ML apps
- **Monitor:** Check Render dashboard for memory usage

### Slow initial load
- **Cause:** Models loaded on first request
- **Fix:** Models are preloaded at startup (see `app.py`). If still slow, first request takes ~5-10s

## 📊 Monitoring

In your Render dashboard:
- **Metrics:** CPU, memory, request count
- **Logs:** Real-time application logs
- **Events:** Deployment history

## 🔐 Security Notes

1. **Flask Secret Key:** Generated by Render automatically
2. **Passwords:** Hashed with Argon2 (already in code)
3. **HTTPS:** Enabled by default on Render
4. **Database:** No public access (SQLite is local)

## 📝 Updates & Redeployment

To update your app:

1. **Make changes locally**
   ```bash
   git add .
   git commit -m "Update features"
   git push origin master
   ```

2. **Render auto-redeploys** (or click Manual Deploy)

3. **Database reset:** Each deployment creates fresh DB
   - To preserve data, migrate to PostgreSQL (see above)

## 🆘 Need Help?

- **Render Docs:** https://render.com/docs
- **Flask Docs:** https://flask.palletsprojects.com/
- **Check Render Logs:** Your dashboard → Logs section shows detailed errors

---

## Summary Checklist

- [ ] `requirements.txt` is Render-optimized (tensorflow-cpu, opencv-python-headless)
- [ ] `Procfile` exists and uses Gunicorn
- [ ] `.gitignore` excludes venv and __pycache__
- [ ] `static/css/output.css` is committed (pre-built Tailwind)
- [ ] ML models in `ml_model/` are committed
- [ ] `app.py` updated for production (host, port, debug=False)
- [ ] Repository pushed to GitHub
- [ ] Web Service created on Render
- [ ] Environment variables set
- [ ] Deployment successful
- [ ] App accessible at your Render URL

---

**Happy deploying!** 🚀

For questions or issues, check the Render dashboard logs first—they provide detailed error messages.
