# 🛰️ Cloud Removal AI v5.0
### Advanced Satellite Imagery Processing with GANs & Gemini AI

**Cloud Removal AI v5.0** is an advanced AI-powered application designed to remove clouds from satellite imagery using Deep Learning and Computer Vision techniques. Beyond cloud removal, it offers 3D terrain reconstruction, Land Use & Land Cover (LULC) classification, and a multilingual AI chatbot assistant.

---

## ⚠️ IMPORTANT: Model Weights Download
> **Note:** The trained model files exceed GitHub's 100MB file size limit. You **MUST** download them manually to run this application.

| File Name | Description | Size | Download Link |
|-----------|-------------|------|---------------|
| `gen_epoch_10.pth` | Generator Model (Required) | ~180MB | [**DOWNLOAD HERE**](https://drive.google.com/drive/folders/1buk4CFEgciR-ddD_ImkHbjAmjAYSY7kB?usp=sharing) |
| `disc_epoch_10.pth` | Discriminator Model (Optional) | ~180MB | [**DOWNLOAD HERE**](https://drive.google.com/drive/folders/1buk4CFEgciR-ddD_ImkHbjAmjAYSY7kB?usp=sharing) |

**Setup:**
1. Download the `.pth` files from the link above.
2. Place them directly inside the `src/` directory of this repository.

---

## ✨ Key Features (v5.0)

### ☁️ Cloud Removal Core
* **Architecture:** Custom GAN-based UNet (Generator + Discriminator).
* **Input:** 4 Channels (RGB + SAR data) for superior cloud penetration.
* **Output:** Clean, cloud-free RGB imagery.

### 🏔️ 3D Terrain Reconstruction
* **Interactive Visualization:** Fully interactive 3D maps powered by Plotly.
* **Controls:** Rotate, zoom, and pan capabilities.
* **Customization:** Adjustable height exaggeration (0.5x - 5x) to analyze terrain depth.

### 🏘️ LULC Classification & Analytics
* **5-Class Segmentation:** Automatically classifies land into:
    * 🌊 Water
    * 🌲 Forest
    * 🏙️ Urban
    * 🏜️ Barren
    * 🌱 Vegetation
* **Analytics:** Provides pie charts showing distribution percentages of each land type.

### 🤖 AI Chatbot Assistant
* **Powered by:** Google Gemini AI.
* **Multilingual Support:** Ask questions in **English** or **Tamil/Tanglish**.
* **Context Aware:** Can analyze the map and answer questions like:
    * *"How much water is in this area?"*
    * *"Evlo thanni irukku?"* (Tamil)

---

## 🏗️ Technical Architecture
1.  **Input:** Cloudy Satellite Image + SAR Data.
2.  **Processing:** GAN-based UNet removes clouds.
3.  **Analysis:** The clean image is processed for LULC, Vegetation Index (VARI), and 3D terrain estimation.
4.  **Output:** Displayed on an interactive Streamlit Dashboard.

---

## 🛠️ Technology Stack

| Category | Tools/Libraries |
|----------|-----------------|
| **Deep Learning** | PyTorch, torchvision, Google Gemini AI |
| **Computer Vision** | OpenCV, PIL/Pillow, scikit-image |
| **Visualization/UI** | Streamlit, Plotly, Matplotlib |
| **Data Processing** | NumPy, Pandas, SciPy |

---

## 🚀 How to Run

### Option 1: Local Installation
```bash
# 1. Clone the repository
git clone https://github.com/9003714553/satelite.git
cd satelite

# 2. Install dependencies
pip install -r requirements.txt

# 3. Place the downloaded .pth files in the 'src/' folder

# 4. Run the app
streamlit run src/app.py
```

### Option 2: Google Colab (Recommended)

1. Upload the project folder to your Google Drive.
2. Open `Cloud_Removal_Colab.ipynb`.
3. Mount Drive and install dependencies.
4. Configure `ngrok` with your auth token for a public URL.
5. Run the cell to launch the app!

---

## 🎯 Use Cases

* **🌾 Agriculture:** Crop monitoring and yield prediction.
* **🏙️ Urban Planning:** Infrastructure mapping.
* **🚨 Disaster Response:** Flood mapping and assessment.
* **🌊 Environmental:** Water detection and monitoring.

---

*Created by [Your Name] - MCA Student*
