# ✅ RENDER DEPLOYMENT QUICK-START CHECKLIST

## 🎯 Pre-Deployment (Local Testing)

- [ ] Navigate to project: `cd "C:\Users\Lenovo\Desktop\mood tracker app"`
- [ ] Pull latest changes: `git pull origin master`
- [ ] Verify `requirements.txt` has gunicorn and tensorflow-cpu
- [ ] Verify `Procfile` exists in root
- [ ] Test locally: `python app.py` (should start with "✅ All ML models loaded")
- [ ] Check no absolute paths in `app.py` (all use `os.path.join()`)
- [ ] Verify `ml_model/` folder exists with model files:
  - `ml_model/Emotion_Dectector/model.h5`
  - `ml_model/Emotion_Dectector/haarcascade_frontalface_default.xml`
  - `ml_model/NLP_Text_Emotion/models/emotion_classifier_pipe_lr_03_jan_2022.pkl`
- [ ] Verify `templates/` folder has all HTML files
- [ ] Verify `static/css/output.css` exists (Tailwind pre-built)
- [ ] Check `.gitignore` excludes `venv/`, `__pycache__/`, `.env`, `instance/app.db`

## 📤 GitHub Commit & Push

```bash
git add .
git commit -m "Render deployment: update app.py, requirements, add Procfile"
git push origin master
```

- [ ] All changes committed locally
- [ ] All changes pushed to GitHub master branch
- [ ] Repository is public or accessible with your GitHub account
- [ ] Verify GitHub shows all files (especially `ml_model/`, `templates/`, `static/`)

## 🌐 Render Dashboard Setup

1. **Create Web Service**
   - [ ] Go to https://render.com → Click **New** → **Web Service**
   - [ ] Connect GitHub → Select **MindSpace** repository
   - [ ] Set **Branch** to `master`
   - [ ] Set **Region** to `Oregon` (or closest to you)
   - [ ] Set **Plan** to `Free`

2. **Build & Start Commands**
   - [ ] Leave **Build Command** blank (defaults to `pip install -r requirements.txt`)
   - [ ] Leave **Start Command** blank (Procfile will be used)

3. **Environment Variables**
   - [ ] Click **Add Environment Variable**
   - [ ] Add `FLASK_ENV` = `production`
   - [ ] Add `APP_SECRET_KEY` = (generate random string, e.g., `openssl rand -hex 24`)

4. **Create Service**
   - [ ] Click **Create Web Service**
   - [ ] Wait 3-5 minutes for build to complete

## 📊 Deployment Verification

- [ ] Check **Logs** tab for build status (look for "✅ All ML models loaded")
- [ ] If build fails, check error message and fix locally → commit → push → auto-redeploy
- [ ] Once deployment successful, visit your app URL
- [ ] Test: Register user → Log mood → Write journal → Check predictions
- [ ] Verify CSS loads (page should be styled, not plain white)

## 🔍 Troubleshooting (If Needed)

- [ ] **Build fails**: Check Logs for error → Update `requirements.txt` → Commit & push
- [ ] **Model not found**: Ensure `ml_model/` files are committed to git
- [ ] **Timeout**: Models are preloaded at startup (should not timeout)
- [ ] **CSS broken**: Verify `static/css/output.css` is committed
- [ ] **Port error**: Procfile already handles PORT from Render environment

## 📝 Post-Deployment

- [ ] Bookmark your app URL: `https://mindspace-mood-tracker.onrender.com`
- [ ] Monitor **Metrics** tab for CPU/memory usage
- [ ] Check **Logs** periodically for errors
- [ ] If updating code: commit → push → Render auto-redeploys

---

## 📁 Quick File Reference

| File | Status | Location |
|------|--------|----------|
| `requirements.txt` | ✅ Updated | Root |
| `app.py` | ✅ Updated | Root |
| `Procfile` | ✅ Created | Root |
| `render.yaml` | ✅ Created | Root (optional) |
| `.gitignore` | ✅ Updated | Root |
| `README_deploy.md` | ✅ Created | Root |
| `RENDER_DEPLOYMENT_GUIDE.md` | ✅ Created | Root |
| `DEPLOYMENT_FILES_SUMMARY.md` | ✅ Created | Root |
| `ml_model/` | ✅ Committed | Root |
| `templates/` | ✅ Committed | Root |
| `static/` | ✅ Committed | Root |

---

## 🚀 YOU'RE READY!

All necessary files are created and updated. Follow the Render Dashboard Setup steps above, and your MindSpace app will be live in minutes!

**Still need help?** Read `README_deploy.md` for detailed step-by-step instructions.

