 # 🚀 Deployment Guide - Streamlit Cloud

Deploy your Cloud Removal AI v5.0 to the internet in 5-10 minutes!

---

## 📋 Prerequisites

1. ✅ GitHub account (free)
2. ✅ Streamlit Cloud account (free) - https://streamlit.io/cloud
3. ✅ Your project code (already ready!)

---

## 🎯 Step-by-Step Deployment

### Step 1: Prepare Your Project

**Create `.gitignore` file** (to avoid uploading large files):

```
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/

# Model checkpoints (too large for GitHub)
*.pth
*.pt
*.ckpt

# Data
data/
dataset/
*.zip

# IDE
.vscode/
.idea/
*.swp

# OS
.DS_Store
Thumbs.db
```

**Important:** Your model checkpoint files (`gen_epoch_5.pth`, `disc_epoch_5.pth`) are too large for GitHub (188MB+). You have two options:

**Option A: Use Mock Data Only**
- The app will work with example/mock data
- No cloud removal, but all other features work
- Best for demo purposes

**Option B: Use Git LFS (Large File Storage)**
- Allows uploading large model files
- Requires Git LFS setup
- More complex but full functionality

---

### Step 2: Upload to GitHub

#### 2.1 Initialize Git (if not already done)

```bash
cd "c:\Users\hp\Downloads\projects AI\Satellite-Imagery-Cloud-Removal\High-Resolution-Satellite-Imagery-Cloud-Removal"
git init
```

#### 2.2 Create `.gitignore`

Create the `.gitignore` file with the content above.

#### 2.3 Add and Commit Files

```bash
git add .
git commit -m "Initial commit: Cloud Removal AI v5.0 with 3D Terrain, LULC, and AI Chatbot"
```

#### 2.4 Create GitHub Repository

1. Go to https://github.com/new
2. Repository name: `satellite-cloud-removal-ai`
3. Description: `Advanced satellite imagery cloud removal with 3D terrain reconstruction, land cover classification, and AI chatbot`
4. Make it **Public** (required for free Streamlit Cloud)
5. Click **"Create repository"**

#### 2.5 Push to GitHub

```bash
git remote add origin https://github.com/YOUR_USERNAME/satellite-cloud-removal-ai.git
git branch -M main
git push -u origin main
```

Replace `YOUR_USERNAME` with your GitHub username.

---

### Step 3: Deploy on Streamlit Cloud

#### 3.1 Sign Up for Streamlit Cloud

1. Go to https://share.streamlit.io/
2. Click **"Sign up"**
3. Sign in with your GitHub account
4. Authorize Streamlit to access your repositories

#### 3.2 Create New App

1. Click **"New app"**
2. Select your repository: `satellite-cloud-removal-ai`
3. Branch: `main`
4. Main file path: `src/app.py`
5. App URL: Choose a custom name (e.g., `cloud-removal-ai`)
6. Click **"Deploy!"**

#### 3.3 Wait for Deployment

- Streamlit Cloud will install dependencies from `requirements.txt`
- This takes 2-5 minutes
- You'll see a progress indicator

---

### Step 4: Share Your App! 🎉

Once deployed, you'll get a URL like:

```
https://cloud-removal-ai.streamlit.app
```

**Share this link with:**
- 📧 Recruiters
- 👥 Friends
- 💼 LinkedIn
- 📱 Portfolio website

---

## ⚠️ Important Notes

### Model Checkpoint Issue

Since your model files are too large for GitHub, the deployed app will use **mock data** by default. This means:

✅ **Will work:**
- 3D Terrain Reconstruction
- LULC Classification
- AI Chatbot
- All UI features

❌ **Won't work:**
- Actual cloud removal (will use random/mock output)

### Solution: Use Smaller Model or External Storage

**Option 1: Train a smaller model**
- Reduce model size
- Use quantization
- Upload to GitHub

**Option 2: Use external storage**
- Upload model to Google Drive
- Download in app using `gdown`
- Add to `requirements.txt`: `gdown`

**Option 3: Demo mode only**
- Keep as-is
- Mention in README that it's a demo
- Show screenshots/videos of local results

---

## 🎨 Customization

### Update App Title

In `src/app.py`, change:
```python
st.set_page_config(
    page_title="Your Name - Cloud Removal AI",
    page_icon="🛰️",
)
```

### Add Your Info

Create `src/sidebar_info.py`:
```python
import streamlit as st

def show_developer_info():
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 👨‍💻 Developer")
    st.sidebar.markdown("**Your Name**")
    st.sidebar.markdown("[LinkedIn](your-linkedin-url)")
    st.sidebar.markdown("[GitHub](your-github-url)")
```

Then in `app.py`:
```python
from sidebar_info import show_developer_info
show_developer_info()
```

---

## 📊 Monitoring

After deployment:
- View app analytics in Streamlit Cloud dashboard
- See visitor count
- Monitor errors
- Check resource usage

---

## 🔄 Updates

To update your deployed app:

```bash
# Make changes to your code
git add .
git commit -m "Update: Added new feature"
git push
```

Streamlit Cloud will **automatically redeploy** within 1-2 minutes!

---

## 🆘 Troubleshooting

### "App is sleeping"
- Free tier apps sleep after inactivity
- Click to wake up (takes ~30 seconds)

### "Requirements installation failed"
- Check `requirements.txt` for typos
- Remove incompatible packages
- Check Streamlit Cloud logs

### "Module not found"
- Ensure all imports are in `requirements.txt`
- Check file paths are relative, not absolute

---

## 🎯 Next Steps After Deployment

1. **Add to Portfolio**
   - Create a project card
   - Add screenshots
   - Link to live demo

2. **Share on LinkedIn**
   ```
   🚀 Excited to share my latest project: Cloud Removal AI v5.0!
   
   Features:
   🏔️ 3D Terrain Reconstruction
   🏘️ Land Cover Classification (5 classes)
   🤖 AI Chatbot (Tamil/Tanglish support!)
   
   Try it live: [your-app-url]
   GitHub: [your-repo-url]
   
   #MachineLearning #ComputerVision #AI #Streamlit
   ```

3. **Create Demo Video**
   - Record screen while using app
   - Upload to YouTube
   - Add to README

---

**Good luck with your deployment! 🚀**
