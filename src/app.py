import streamlit as st
import torch
import numpy as np
import os
from data_loader import get_dataloader

# Attempt to import model, handle failure
try:
    from models import UNetGenerator
    MODEL_AVAILABLE = True
except ImportError:
    MODEL_AVAILABLE = False

st.set_page_config(page_title="Cloud Removal", layout="wide")

st.title("High-Resolution Satellite Imagery Cloud Removal")

def to_image(tensor):
    """Convert tensor to numpy for display."""
    img = tensor.permute(1, 2, 0).cpu().numpy()
    # Handle if data is not roughly 0-1. Mock data is randn.
    # Normalize to 0-1 for display
    img = (img - img.min()) / (img.max() - img.min())
    return img

st.sidebar.header("Configuration")
use_mock = st.sidebar.checkbox("Use Mock Data", value=True)

if st.button("Load & Run"):
    with st.spinner("Loading data..."):
        loader = get_dataloader(batch_size=1, split='val', mock_data=use_mock)
        batch = next(iter(loader))
        
        cloudy = batch['cloudy'][0]
        sar = batch['sar'][0]
        clear = batch['clear'][0]
        
    st.subheader("Input Images")
    c1, c2, c3 = st.columns(3)
    
    c1.image(to_image(cloudy), caption="Cloudy (Optical)", use_container_width=True)
    c2.image(to_image(sar), caption="SAR (Radar)", use_container_width=True)
    c3.image(to_image(clear), caption="Ground Truth (Clear)", use_container_width=True)
    
    st.divider()
    
    st.subheader("Model Prediction")
    
    if MODEL_AVAILABLE:
        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            gen = UNetGenerator().to(device)
            chk_path = "../gen_epoch_10.pth"
            if os.path.exists(chk_path):
                try:
                    checkpoint = torch.load(chk_path, map_location=device)
                    # Handle dictionary or direct state dict
                    if "state_dict" in checkpoint:
                         gen.load_state_dict(checkpoint["state_dict"])
                    else:
                         gen.load_state_dict(checkpoint)
                    st.success(f"Loaded checkpoint: {chk_path}")
                except Exception as e:
                    st.warning(f"Could not load checkpoint '{chk_path}': {e}. Using random weights.")
            else:
                st.warning(f"Checkpoint not found at {chk_path}, using random weights.")
                
            gen.eval()
            with torch.no_grad():
                gen_input = torch.cat([cloudy.unsqueeze(0), sar.unsqueeze(0)], dim=1).to(device)
                fake_clear = gen(gen_input)
                
            st.image(to_image(fake_clear[0]), caption="Generated Clear Image", use_container_width=True)
                
        except Exception as e:
            st.error(f"Error running model: {e}")
    else:
        st.error("Model class 'UNetGenerator' not found in `models.py`. Cannot run inference.")
        st.info("The project structure seems to have a mismatch between `predict.py` (Cloud Removal) and `models.py` (Change Detection/SiameseNetwork).")

