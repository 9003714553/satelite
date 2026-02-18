# Streamlit Cloud Deployment Guide

Complete step-by-step guide to deploy your Satellite Imagery Cloud Removal AI to Streamlit Cloud.

---

## ✅ Pre-Deployment Checklist

Your project already has:
- ✅ `src/app.py` (691 lines - main Streamlit application)
- ✅ `requirements.txt` (all dependencies listed)
- ✅ Model files uploaded to Hugging Face
- ✅ Complete documentation

---

## 🚀 Deployment Steps

### Step 1: Push to GitHub

Your project needs to be on GitHub for Streamlit Cloud to access it.

**Option A: Create New Repository (If not on GitHub yet)**

1. Go to: https://github.com/new
2. Repository name: `satellite-cloud-removal`
3. Description: `AI-powered satellite imagery cloud removal with 3D terrain and land cover analysis`
4. Visibility: **Public** (required for free Streamlit Cloud)
5. Click **"Create repository"**

**Option B: Initialize Git in Your Project**

Open Command Prompt in your project folder:

```bash
cd "c:\Users\hp\Downloads\projects AI\Satellite-Imagery-Cloud-Removal\High-Resolution-Satellite-Imagery-Cloud-Removal"

# Initialize Git
git init

# Add all files
git add .

# Commit
git commit -m "Initial commit: Satellite Cloud Removal AI v5.0"

# Connect to GitHub (replace YOUR-USERNAME)
git remote add origin https://github.com/VIJAYarajan03/satellite-cloud-removal.git

# Push to GitHub
git push -u origin main
```

---

### Step 2: Handle Large Model Files

Since your model files are ~195 MB, you have two options:

**Option A: Use Git LFS (Recommended)**

```bash
# Install Git LFS
git lfs install

# Track .pth files
git lfs track "*.pth"

# Add .gitattributes
git add .gitattributes

# Add model files
git add gen_epoch_5.pth disc_epoch_5.pth disc_epoch_10.pth

# Commit and push
git commit -m "Add model files with Git LFS"
git push
```

**Option B: Download from Hugging Face (Easier for Streamlit Cloud)**

Modify `src/app.py` to download models from Hugging Face on first run:

```python
from huggingface_hub import hf_hub_download
import os

# At the top of app.py, add this function
def download_models():
    """Download models from Hugging Face if not present"""
    model_files = ['gen_epoch_5.pth', 'disc_epoch_5.pth']
    repo_id = "VIJAYarajan03/satellite-cloud-removal"
    
    for filename in model_files:
        if not os.path.exists(filename):
            print(f"Downloading {filename} from Hugging Face...")
            hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                local_dir=".",
                local_dir_use_symlinks=False
            )
    print("Models ready!")

# Call before loading models
download_models()
```

---

### Step 3: Update requirements.txt

Make sure `requirements.txt` includes:

```
streamlit
torch
torchvision
Pillow
numpy
matplotlib
plotly
opencv-python-headless
scipy>=1.10.0
pandas>=2.0.0
streamlit-image-comparison
streamlit-folium
folium
fpdf
streamlit-image-zoom
google-generativeai>=0.3.0
huggingface_hub
```

---

### Step 4: Create .streamlit/config.toml

Create folder `.streamlit` and file `config.toml`:

```toml
[theme]
primaryColor = "#FF4B4B"
backgroundColor = "#0E1117"
secondaryBackgroundColor = "#262730"
textColor = "#FAFAFA"
font = "sans serif"

[server]
maxUploadSize = 200
enableXsrfProtection = false
```

---

### Step 5: Deploy to Streamlit Cloud

1. **Go to Streamlit Cloud**
   - Visit: https://share.streamlit.io/
   - Click **"Sign up"** or **"Sign in"** (use GitHub account)

2. **Create New App**
   - Click **"New app"**
   - Repository: `VIJAYarajan03/satellite-cloud-removal`
   - Branch: `main`
   - Main file path: `src/app.py`
   - App URL: Choose a custom name like `satellite-cloud-removal`

3. **Advanced Settings (Optional)**
   - Python version: 3.9 or 3.10
   - Add secrets if needed (for API keys)

4. **Deploy!**
   - Click **"Deploy"**
   - Wait 5-10 minutes for build
   - Your app will be live at: `https://YOUR-APP-NAME.streamlit.app`

---

## 🔧 Common Issues & Solutions

### Issue 1: Model Files Too Large

**Solution:** Use Option B (download from Hugging Face) instead of including in Git.

### Issue 2: Build Timeout

**Solution:** 
- Reduce dependencies in `requirements.txt`
- Use `opencv-python-headless` instead of `opencv-python`
- Remove unused libraries

### Issue 3: Memory Errors

**Solution:**
- Streamlit Cloud free tier has 1GB RAM
- Load models only when needed
- Clear cache after processing

### Issue 4: Import Errors

**Solution:**
- Make sure all imports are in `requirements.txt`
- Use exact versions that work locally

---

## 📊 After Deployment

### Monitor Your App

- Check logs in Streamlit Cloud dashboard
- Monitor resource usage
- Track visitor analytics

### Share Your App

Add badges to your README:

```markdown
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://YOUR-APP-NAME.streamlit.app)
[![Hugging Face](https://img.shields.io/badge/🤗%20Hugging%20Face-Model-yellow)](https://huggingface.co/VIJAYarajan03/satellite-cloud-removal)
```

### Update Your App

Any push to GitHub `main` branch will automatically redeploy!

```bash
# Make changes
git add .
git commit -m "Update feature"
git push

# Streamlit Cloud auto-redeploys in ~2 minutes
```

---

## 🎯 Quick Commands Summary

```bash
# 1. Initialize Git
git init
git add .
git commit -m "Initial commit"

# 2. Push to GitHub
git remote add origin https://github.com/VIJAYarajan03/satellite-cloud-removal.git
git push -u origin main

# 3. Deploy on Streamlit Cloud
# Go to https://share.streamlit.io/ and follow UI

# 4. Update app
git add .
git commit -m "Update"
git push
```

---

## 🌟 Your Live App Will Have:

- ☁️ Cloud removal demo
- 🏔️ 3D terrain visualization
- 🏘️ Land cover classification
- 🤖 AI chatbot (Tamil/English)
- 📊 Analytics and reports
- 🗺️ GPS mapping
- 📥 Batch processing

**All accessible via a single URL!**

---

## 📝 Next Steps

1. Push code to GitHub
2. Deploy to Streamlit Cloud
3. Share link on LinkedIn/Portfolio
4. Add to resume as live project demo

**Good luck with deployment!** 🚀
