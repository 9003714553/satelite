# High-Resolution Satellite Imagery Change Detection

## Overview
This project implements a **change detection system** for high-resolution satellite imagery using a **Siamese Neural Network**. The system is designed to identify significant changes (e.g., new infrastructure, deforestation, damage) between two temporal images ($t_1$ and $t_2$) of the same location.

## Architecture
- **Input**: Two co-registered satellite images ($t_1, t_2$).
- **Encoder**: A Siamese network with shared weights (ResNet18 backbone) extracts feature maps from both images.
- **Fusion**: Feature maps are compared (e.g., using absolute difference or concatenation).
- **Decoder**: A semantic segmentation head (like FCN or U-Net decoder) produces a pixel-wise binary change mask.

## Installation
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
### Inference
To detect changes between two images:
```bash
python src/predict.py --before images/time1.png --after images/time2.png --output output.png
```

### Training
To train the model on a dataset:
```bash
python src/train.py --data_dir /path/to/dataset
```
