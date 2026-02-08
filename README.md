# 🛰️ Cloud Removal AI v5.0 - Advanced Edition

Advanced satellite imagery cloud removal with **3D Terrain Reconstruction**, **Land Cover Classification**, and **AI-Powered Analysis**.

---

## 🎉 What's New in v5.0

### 🏔️ 3D Terrain Reconstruction
- **Interactive 3D visualization** of satellite imagery
- **Adjustable height exaggeration** (0.5x - 5x)
- **Two estimation methods:** Brightness-based and gradient-based
- **Full camera controls:** Rotate, zoom, pan with mouse

### 🏘️ Land Use & Land Cover (LULC) Classification
- **5 Land Cover Classes:**
  - 💧 Water (Rivers, lakes, oceans)
  - 🌲 Forest (Dense vegetation)
  - 🏙️ Urban (Buildings, roads)
  - 🏜️ Barren (Bare soil, rocks)
  - 🌾 Vegetation (Crops, grassland)
- **Color-coded maps** with interactive legend
- **Pie charts** showing distribution percentages
- **Spatial analysis** (North, South, East, West)

### 🤖 AI Chatbot Assistant
- **Natural language queries** about your map
- **Supported questions:**
  - "How much water is in this area?"
  - "Where is the urban area located?"
  - "Is there more forest or vegetation?"
  - "Tell me about this map"
- **Tamil/Tanglish support** (e.g., "Evlo thanni irukku?")
- **Chat history** with last 5 conversations

---

## 🚀 Quick Start

### Installation

```bash
# Navigate to project directory
cd "c:\Users\hp\Downloads\projects AI\Satellite-Imagery-Cloud-Removal\High-Resolution-Satellite-Imagery-Cloud-Removal"

# Install dependencies
pip install -r requirements.txt
```

### Run the App

```bash
streamlit run src/app.py
```

The app will open in your browser at `http://localhost:8501`

---

## 📖 Usage Guide

### 1️⃣ Process an Image
1. Go to **🚀 Workspace** tab
2. Click **Load New** for example data, or upload your own satellite image
3. Wait for cloud removal processing

### 2️⃣ View in 3D
1. Navigate to **🏔️ 3D Terrain** tab
2. Adjust controls:
   - **Height Exaggeration:** 2.0 (recommended)
   - **Method:** brightness or gradient
   - **Resolution:** 0.75 for balanced performance
3. Interact with the 3D plot using your mouse

### 3️⃣ Analyze Land Cover
1. Navigate to **🏘️ Land Cover (LULC)** tab
2. View color-coded classification map
3. Check pie chart for distribution
4. Ask questions to the AI chatbot

---

## 📦 New Dependencies

The following packages were added in v5.0:

```
plotly>=5.14.0              # Interactive 3D visualization
scipy>=1.10.0               # Gaussian filtering
numpy-stl>=3.0.0            # STL export (optional)
segmentation-models-pytorch>=0.3.3  # Future ML models
pandas>=2.0.0               # Data analysis
google-generativeai>=0.3.0  # Optional Gemini API
```

---

## 🗂️ Project Structure

```
src/
├── app.py                  # Main Streamlit application (v5.0)
├── terrain_3d.py          # 🆕 3D terrain reconstruction
├── lulc_classifier.py     # 🆕 Land cover classification
├── chatbot.py             # 🆕 AI chatbot assistant
├── models.py              # UNet model definition
├── train.py               # Training script
├── predict.py             # Prediction script
├── dataset.py             # Dataset loader
└── data_loader.py         # Data loading utilities
```

---

## 🎨 Features Overview

### Previous Features (v4.0)
- ☁️ Cloud removal using GAN-based UNet
- 🌱 Vegetation health index (VARI)
- 🛣️ Infrastructure/road extraction
- 🔍 Magic lens (zoom tool)
- 📄 PDF report generation
- 📦 Batch processing
- 🗺️ GPS geo-tagging
- 📊 Analytics dashboard

### New Features (v5.0)
- 🏔️ 3D terrain reconstruction
- 🏘️ LULC classification
- 🤖 AI chatbot assistant

---

## 🧪 Example Queries for Chatbot

Try asking these questions:

**English:**
- "How much water is in this area?"
- "Where is the forest located?"
- "Is there more urban or vegetation?"
- "Tell me about this map"

**Tamil/Tanglish:**
- "Evlo thanni irukku?"
- "Kaadu enga irukku?"
- "Veedu athigama illa pachai athigama?"

---

## 🔧 Technical Details

### 3D Terrain Algorithm
1. Convert image to grayscale (luminance formula)
2. Apply Gaussian smoothing (σ=2.0)
3. Normalize to [0, 1] range
4. Generate meshgrid for X, Y coordinates
5. Apply height exaggeration to Z values
6. Render with Plotly Surface plot

### LULC Classification Method
- **Rule-based approach** using color indices:
  - Vegetation Index: `(Green - Red) / (Green + Red)`
  - Water Index: `Blue - (Red + Green) / 2`
  - Urban Index: `Brightness - Vegetation Index`
- **Post-processing:** Median blur (5x5 kernel)

### Chatbot Architecture
- **Intent parsing:** Keyword-based classification
- **Entity extraction:** Land cover class detection
- **Response generation:** Template-based with statistics
- **Optional:** Gemini API for advanced queries

---

## 🎯 Performance Tips

1. **3D Terrain:**
   - Use lower resolution (0.5x) for large images
   - Brightness method is faster than gradient
   - Reduce height exaggeration for subtle terrain

2. **LULC Classification:**
   - Works best on clear, cloud-free images
   - Results cached in session state
   - Instant re-rendering on tab switch

3. **Chatbot:**
   - Rule-based responses are instant
   - Gemini API adds ~2-3 second latency
   - Chat history limited to 5 messages

---

## 📝 License

This project is for educational and research purposes.

---

## 🙏 Acknowledgments

- **Streamlit** - Web framework
- **Plotly** - 3D visualization
- **PyTorch** - Deep learning backend
- **OpenCV** - Image processing

---

## 📧 Support

For issues or questions, please check the walkthrough documentation in the artifacts folder.

---

**Enjoy exploring your satellite imagery in 3D! 🚀**
