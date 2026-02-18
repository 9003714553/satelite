# 🤗 Hugging Face Upload Guide

Complete step-by-step guide to upload your 3.55 GB Satellite Imagery Cloud Removal project to Hugging Face.

---

## 📋 Prerequisites

1. ✅ Hugging Face account (free) - https://huggingface.co/join
2. ✅ Email confirmed
3. ✅ Project ready (3.55 GB)

---

## 🚀 Step-by-Step Process

### Step 1: Install Hugging Face CLI

Open Command Prompt and run:

```bash
pip install huggingface_hub
```

**Expected Output:**
```
Successfully installed huggingface_hub-x.x.x
```

---

### Step 2: Login to Hugging Face

Run this command:

```bash
huggingface-cli login
```

**What happens:**
1. It will ask for your **Access Token**
2. Go to: https://huggingface.co/settings/tokens
3. Click **"New token"**
4. Name: `upload-token`
5. Type: **Write** (important!)
6. Click **"Generate"**
7. Copy the token
8. Paste in Command Prompt (won't show on screen, just press Enter)

**Expected Output:**
```
Token is valid (permission: write).
Your token has been saved to C:\Users\hp\.huggingface\token
Login successful
```

---

### Step 3: Create Repository on Hugging Face

**Option A: Via Website (Easier)**
1. Go to: https://huggingface.co/new
2. Repository name: `satellite-cloud-removal` (or your choice)
3. Type: **Model** (recommended) or **Dataset**
4. Visibility: **Public** (for free tier)
5. Click **"Create repository"**

**Option B: Via Command Line**
```bash
huggingface-cli repo create satellite-cloud-removal --type model
```

**Your Repository URL will be:**
```
https://huggingface.co/your-username/satellite-cloud-removal
```

---

### Step 4: Configure Upload Script

1. Open `upload_to_huggingface.py` in this folder
2. Update line 13:
   ```python
   REPO_ID = "your-username/satellite-cloud-removal"
   ```
   Replace `your-username` with your actual Hugging Face username

**Example:**
```python
REPO_ID = "prakash/satellite-cloud-removal"
```

---

### Step 5: Run Upload Script

In Command Prompt, navigate to project folder:

```bash
cd "c:\Users\hp\Downloads\projects AI\Satellite-Imagery-Cloud-Removal\High-Resolution-Satellite-Imagery-Cloud-Removal"
```

Then run:

```bash
python upload_to_huggingface.py
```

**Expected Process:**
```
============================================================
🚀 Hugging Face Upload Script
============================================================
📁 Folder: c:\Users\hp\Downloads\projects AI\...
🎯 Repository: your-username/satellite-cloud-removal
📦 Type: model
💾 Size: ~3.55 GB
============================================================

⏳ Upload starting... This will take some time (3.55 GB)
⚠️  Do NOT close this window or interrupt the process!
------------------------------------------------------------
Uploading files: 100%|████████████████| 45/45 [15:23<00:00, ...]
============================================================
✅ SUCCESS! All files uploaded to Hugging Face!
============================================================
🔗 View your project at: https://huggingface.co/your-username/satellite-cloud-removal
============================================================
```

---

## ⏱️ Upload Time Estimate

| Internet Speed | Estimated Time |
|----------------|----------------|
| 10 Mbps | ~50-60 minutes |
| 20 Mbps | ~25-30 minutes |
| 50 Mbps | ~10-15 minutes |
| 100 Mbps | ~5-8 minutes |

**Important:** Don't close the terminal during upload!

---

## 📁 What Gets Uploaded

Your entire project structure:
```
✅ src/ (all Python files)
✅ gen_epoch_5.pth (188 MB model)
✅ disc_epoch_5.pth (8 MB model)
✅ requirements.txt
✅ README.md
✅ DEPLOYMENT.md
✅ data/ (if present)
✅ All other files
```

**Total Size:** ~3.55 GB

---

## 🔧 Troubleshooting

### Error: "Repository not found"
**Solution:** Make sure you created the repository first (Step 3)

### Error: "Invalid token"
**Solution:** 
1. Check token has **Write** permission
2. Re-run `huggingface-cli login`

### Error: "Connection timeout"
**Solution:**
1. Check internet connection
2. Try again - upload will resume from where it stopped

### Upload stuck at certain file
**Solution:**
1. Press `Ctrl+C` to cancel
2. Run the script again - it will skip already uploaded files

---

## 📝 After Upload

### Update README on Hugging Face

Add this to your repository's README:

```markdown
# 🛰️ Satellite Imagery Cloud Removal AI v5.0

Advanced satellite imagery cloud removal with 3D terrain reconstruction, land cover classification, and AI chatbot.

## Features
- ☁️ Cloud removal using GAN-based UNet
- 🏔️ 3D terrain reconstruction
- 🏘️ Land cover classification (5 classes)
- 🤖 AI chatbot (Tamil/English)
- 🌱 Vegetation health monitoring
- 🛣️ Infrastructure extraction

## Quick Start

```bash
# Clone repository
git clone https://huggingface.co/your-username/satellite-cloud-removal

# Install dependencies
pip install -r requirements.txt

# Run application
streamlit run src/app.py
```

## Model Details
- Architecture: UNet Generator (GAN)
- Input: 4 channels (RGB + SAR)
- Output: 3 channels (RGB)
- Size: 188 MB
- PSNR: 25-30 dB
```

---

## 🎯 Next Steps

After successful upload:

1. **Share your project:**
   - LinkedIn: "Just uploaded my AI project to Hugging Face! 🚀"
   - Twitter: Tag @huggingface
   - Portfolio: Add the Hugging Face link

2. **Add to README.md badges:**
   ```markdown
   [![Hugging Face](https://img.shields.io/badge/🤗%20Hugging%20Face-Model-yellow)](https://huggingface.co/your-username/satellite-cloud-removal)
   ```

3. **Create a Model Card:**
   - Add usage examples
   - Include sample outputs
   - Document limitations

---

## 📊 Repository Statistics

After upload, you'll get:
- ⭐ Star count
- 👁️ View count
- 📥 Download count
- 🔄 Clone count

---

## ✅ Checklist

Before uploading, make sure:
- [ ] Hugging Face account created
- [ ] Email confirmed
- [ ] `huggingface_hub` installed
- [ ] Logged in via CLI
- [ ] Repository created
- [ ] `upload_to_huggingface.py` configured
- [ ] Stable internet connection
- [ ] Enough time for upload (check estimate above)

---

**Good luck with your upload! 🚀**

For issues, check: https://huggingface.co/docs/huggingface_hub
