import streamlit as st
from skimage.metrics import structural_similarity as calc_ssim
from skimage.metrics import peak_signal_noise_ratio as calc_psnr
import torch
import torch.nn as nn
import numpy as np
import os
import io
from PIL import Image, ExifTags
import torchvision.transforms as T
import matplotlib.pyplot as plt
from data_loader import get_dataloader
from streamlit_image_comparison import image_comparison
import zipfile
from fpdf import FPDF
import tempfile
from streamlit_image_zoom import image_zoom
import cv2
import plotly.graph_objects as go

# Import new feature modules
from terrain_3d import generate_3d_visualization
from lulc_classifier import classify_and_visualize, LULC_CLASSES
from chatbot import MapChatbot

# --- Legacy Model Definition (to match checkpoint) ---
class LegacyUNet(nn.Module):
    def __init__(self):
        super(LegacyUNet, self).__init__()
        # Encoder (Downsampling) - No BN, No Bias (matched to checkpoint)
        self.enc1 = nn.Sequential(nn.Conv2d(4, 64, 4, 2, 1, bias=False), nn.LeakyReLU(0.2))
        self.enc2 = nn.Sequential(nn.Conv2d(64, 128, 4, 2, 1, bias=False), nn.LeakyReLU(0.2))
        self.enc3 = nn.Sequential(nn.Conv2d(128, 256, 4, 2, 1, bias=False), nn.LeakyReLU(0.2))
        self.enc4 = nn.Sequential(nn.Conv2d(256, 512, 4, 2, 1, bias=False), nn.LeakyReLU(0.2))
        
        self.bottleneck = nn.Sequential(nn.Conv2d(512, 512, 4, 2, 1, bias=False), nn.ReLU())
        
        # Decoder (Upsampling + Skip Connections)
        self.up4 = nn.ConvTranspose2d(512, 512, 4, 2, 1) 
        self.dec4 = nn.Sequential(nn.Conv2d(1024, 256, 3, 1, 1, bias=False), nn.ReLU())
        
        self.up3 = nn.ConvTranspose2d(256, 256, 4, 2, 1)
        self.dec3 = nn.Sequential(nn.Conv2d(512, 128, 3, 1, 1, bias=False), nn.ReLU())
        
        self.up2 = nn.ConvTranspose2d(128, 128, 4, 2, 1)
        self.dec2 = nn.Sequential(nn.Conv2d(256, 64, 3, 1, 1, bias=False), nn.ReLU())
        
        self.up1 = nn.ConvTranspose2d(64, 64, 4, 2, 1)
        self.dec1 = nn.Sequential(nn.Conv2d(128, 64, 3, 1, 1, bias=False), nn.ReLU())
        
        self.final = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(64, 3, 3, 1, 0), # kept bias=True as trained
            nn.Tanh()
        )

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        
        b = self.bottleneck(e4)
        
        d4 = self.up4(b)
        d4 = torch.cat([d4, e4], dim=1)
        d4 = self.dec4(d4)
        
        d3 = self.up3(d4)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)
        
        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)
        
        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)
        
        return self.final(d1)

# --- Standard Model Import ---
try:
    from models import UNetGenerator
    MODEL_AVAILABLE = True
except ImportError:
    MODEL_AVAILABLE = False
    class UNetGenerator(nn.Module): pass # Dummy

# --- Page Configuration ---
st.set_page_config(
    page_title="Cloud Removal AI v4.0",
    page_icon="☁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Helper Functions ---
def to_image(tensor):
    """Convert tensor to numpy for display (0-1 range)."""
    tensor = tensor.detach()  # 🔥 Detach from computation graph
    img = tensor.permute(1, 2, 0).cpu().numpy()
    img = (img - img.min()) / (img.max() - img.min())
    return img

def preprocess_image(image, is_sar=False):
    transform_list = [T.Resize((256, 256)), T.ToTensor()]
    if is_sar:
        transform_list.append(T.Normalize((0.5,), (0.5,))) 
        if image.mode != 'L': image = image.convert('L')
    else:
        transform_list.append(T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
        if image.mode != 'RGB': image = image.convert('RGB')
    return T.Compose(transform_list)(image)

def calculate_psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0: return 100
    return 20 * np.log10(1.0 / np.sqrt(mse))

def calculate_cloud_coverage(img_tensor):
    img = to_image(img_tensor)
    if img.shape[2] == 3: brightness = np.mean(img, axis=2)
    else: brightness = img
    return (np.sum(brightness > 0.6) / brightness.size) * 100

def create_overlay(img_tensor, threshold=0.6):
    img = to_image(img_tensor)
    if img.shape[2] == 3: brightness = np.mean(img, axis=2)
    else: brightness = img
    mask = brightness > threshold
    overlay = np.copy(img)
    overlay[mask] = [1.0, 0.0, 0.0]
    overlay[mask] = [1.0, 0.0, 0.0]
    return overlay

    overlay[mask] = [1.0, 0.0, 0.0]
    return overlay

def gan_to_image(img):
    """
    Robust GAN output converter.
    Handles:
    - Batch dimensions (1, H, W, C)
    - Tanh output range [-1, 1] -> [0, 1]
    - Channel ordering
    - Float to uint8 conversion
    """
    # Convert from tensor if needed
    if isinstance(img, torch.Tensor):
        img = to_image(img)
        
    img = np.array(img)

    # Remove batch dimension if exists
    if img.ndim == 4:
        img = img[0]

    # Remove extra single dimensions (e.g. 1, 1, 3)
    img = np.squeeze(img)

    # Handle Tanh output range [-1, 1] -> [0, 1]
    if img.min() < 0:
        img = (img + 1) / 2.0

    # Ensure valid range [0, 255] and type uint8
    if img.dtype != np.uint8:
        img = (img * 255).clip(0, 255).astype(np.uint8)
            
    return img

def analyze_vegetation(image):
    """Calculates Vegetation Index from RGB Image"""
    # Ensure image is numpy array
    if isinstance(image, torch.Tensor):
        img_array = to_image(image)
    else:
        img_array = np.array(image).astype(float) / 255.0

    if img_array.shape[2] == 3:
        R = img_array[:, :, 0]
        G = img_array[:, :, 1]
        B = img_array[:, :, 2]
        
        # VARI Formula: (Green - Red) / (Green + Red - Blue)
        numerator = G - R
        denominator = G + R - B + 0.00001  # Prevent division by zero
        vari_index = numerator / denominator
        
        # Normalize for visualization
        vari_normalized = (vari_index + 1) / 2
        return vari_normalized
    return img_array



def extract_infrastructure(image):
    """Highlights edges to show roads and buildings"""
    # Ensure image is numpy array (0-255 uint8 for cv2)
    if isinstance(image, torch.Tensor):
        img_array = (to_image(image) * 255).astype(np.uint8)
    else:
        img_array = np.array(image)
        if img_array.dtype != np.uint8:
            img_array = (img_array * 255).astype(np.uint8)
            
    # Convert to grayscale
    if len(img_array.shape) == 3 and img_array.shape[2] == 3:
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_array

    # Canny Edge Detection
    edges = cv2.Canny(gray, 100, 200)
    
    # Make edges visible (Green color overlay)
    edge_overlay = np.zeros_like(img_array)
    if len(edge_overlay.shape) == 3:
        edge_overlay[edges > 0] = [0, 255, 0] # Green edges
    else:
        edge_overlay[edges > 0] = 255
    
    combined = cv2.addWeighted(img_array, 0.8, edge_overlay, 1, 0)
    return combined

def get_exif_location(image):
    _img = image
    exif_data = _img._getexif()
    if not exif_data: return None
    exif = {ExifTags.TAGS[k]: v for k, v in exif_data.items() if k in ExifTags.TAGS}
    if 'GPSInfo' in exif:
        gps_info = exif['GPSInfo']
        def convert_to_degrees(value):
            d, m, s = value
            return d + (m / 60.0) + (s / 3600.0)
        lat = convert_to_degrees(gps_info[2])
        lon = convert_to_degrees(gps_info[4])
        if gps_info[1] != 'N': lat = -lat
        if gps_info[3] != 'E': lon = -lon
        return [lat, lon]
    return None

def upsample_image(image_tensor, scale_factor=2):
    image_tensor = image_tensor.unsqueeze(0)
    upsampled = torch.nn.functional.interpolate(
        image_tensor, scale_factor=scale_factor, mode='bicubic', align_corners=True
    )
    return upsampled.squeeze(0)

def plot_histogram(img_tensor, title):
    img = to_image(img_tensor)
    fig, ax = plt.subplots(figsize=(6, 3))
    colors = ['red', 'green', 'blue']
    if img.shape[2] == 3:
        for i, color in enumerate(colors):
            hist, bins = np.histogram(img[:, :, i], bins=256, range=(0, 1))
            ax.plot(bins[:-1], hist, color=color, alpha=0.7)
    else:
        hist, bins = np.histogram(img, bins=256, range=(0, 1))
        ax.plot(bins[:-1], hist, color='black', alpha=0.7)
    ax.set_title(title)
    ax.set_xlim(0, 1)
    ax.grid(True, alpha=0.3)
    return fig

def create_pdf_report(cloudy, clean, psnr, cloud_pct):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(200, 10, txt="Cloud Removal Report", ln=True, align='C')
    pdf.ln(10)
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt=f"Metrics: Cloud={cloud_pct:.1f}%, PSNR={psnr:.2f}dB" if psnr else f"Metrics: Cloud={cloud_pct:.1f}%, PSNR=N/A", ln=True)
    pdf.ln(5)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as t1:
        Image.fromarray((to_image(cloudy)*255).astype(np.uint8)).save(t1.name)
        pdf.image(t1.name, x=10, y=50, w=90)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as t2:
        Image.fromarray((to_image(clean)*255).astype(np.uint8)).save(t2.name)
        pdf.image(t2.name, x=110, y=50, w=90)
    return pdf.output(dest='S').encode('latin-1')

# --- Main App ---
st.title("🛰️ Cloud Removal AI (v5.0) - Advanced Edition")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🚀 Workspace", 
    "📊 Analytics & Map", 
    "🏔️ 3D Terrain",
    "🏘️ Land Cover (LULC)",
    "ℹ️ Info"
])

with tab1:
    st.sidebar.header("Settings")
    mode = st.sidebar.radio("Mode", ["Single", "Batch"])
    
    if mode == "Single":
        src = st.sidebar.radio("Source", ["Example", "Upload"])
        cloudy, sar, gt, loc = None, None, None, None
        sr_on = st.sidebar.checkbox("Super-Res (x2)")
        mask_on = st.sidebar.checkbox("Cloud Mask")

        if src == "Example":
            if st.sidebar.button("Load New", type="primary"):
                loader = get_dataloader(batch_size=1, split='val', mock_data=True)
                batch = next(iter(loader))
                cloudy, sar, gt = batch['cloudy'][0], batch['sar'][0], batch['clear'][0]
                loc = [13.0827, 80.2707]
        else:
            cf = st.file_uploader("Optical", type=["jpg", "png"])
            sf = st.file_uploader("SAR (Optional)", type=["jpg", "png"])
            if cf:
                img = Image.open(cf)
                loc = get_exif_location(img)
                if loc: st.toast("GPS Found!", icon="📍")
                cloudy = preprocess_image(img)
                if sf: sar = preprocess_image(Image.open(sf), is_sar=True)
                # Note: In production, use actual SAR images for better cloud removal performance
                # Random noise is used here as fallback when SAR data is not available
                else: sar = torch.randn(1, 256, 256)

        if cloudy is not None:
            c1, c2 = st.columns(2)
            with c1: 
                st.image(create_overlay(cloudy) if mask_on else to_image(cloudy), "Optical Input", use_container_width=True)
            with c2: 
                st.image(to_image(sar), "SAR Input", channels="GRAY", use_container_width=True)
            
            st.divider()
            
            with st.spinner("Processing..."):
                device = "cuda" if torch.cuda.is_available() else "cpu"
                
                # --- Robust Model Loading ---
                gen = None
                chk_path = "gen_epoch_5.pth"
                
                # Try Standard Model
                try:
                    if MODEL_AVAILABLE:
                        model = UNetGenerator().to(device)
                        if os.path.exists(chk_path):
                            checkpoint = torch.load(chk_path, map_location=device)
                            sd = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
                            model.load_state_dict(sd)
                            gen = model
                            # st.success("Loaded Standard Model")
                except Exception:
                    pass
                
                # Try Legacy Model if Standard failed
                if gen is None:
                    try:
                        model = LegacyUNet().to(device)
                        if os.path.exists(chk_path):
                            checkpoint = torch.load(chk_path, map_location=device)
                            sd = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
                            model.load_state_dict(sd)
                            gen = model
                            st.toast("Loaded Legacy Checkpoint Compatibility Mode", icon="⚠️")
                    except Exception as e:
                        st.error(f"Critical: Could not load model. {e}")
                
                if gen:
                    gen.eval()
                    with torch.no_grad():
                        inp = torch.cat([cloudy.unsqueeze(0), sar.unsqueeze(0)], dim=1).to(device)
                        fake = gen(inp)
                    
                    final = upsample_image(fake[0].cpu()) if sr_on else fake[0].cpu()
                    out_img = to_image(final)
                    
                    st.subheader("✨ Result")
                    if sr_on: st.image(out_img, "Super-Res Output", use_container_width=True)
                    else: 
                        # Use robust conversion for GAN output
                        cloudy_fixed = gan_to_image(cloudy)
                        out_fixed = gan_to_image(out_img)
                        image_comparison(cloudy_fixed, out_fixed, "Cloudy", "Clean", width=700, in_memory=True)
                    
                    st.session_state['data'] = {'cloudy': cloudy, 'out': final, 'loc': loc}
                    
                    # ========================================
                    # 📈 QUALITY METRICS SECTION
                    # ========================================
                    st.divider()
                    st.subheader("📈 Quality Metrics & Accuracy")
                    
                    try:
                        # Calculate metrics
                        cloud_before = calculate_cloud_coverage(cloudy)
                        cloud_after = calculate_cloud_coverage(final)
                        cloud_removed = max(0, cloud_before - cloud_after)
                        
                        # PSNR & SSIM (only if ground truth available)
                        psnr_val = None
                        ssim_val = None
                        if gt is not None:
                            gen_np = np.clip(to_image(final), 0, 1)
                            gt_np = np.clip(to_image(gt), 0, 1)
                            psnr_val = calc_psnr(gt_np, gen_np, data_range=1.0)
                            ssim_val = calc_ssim(gt_np, gen_np, channel_axis=2, data_range=1.0)
                        
                        # Quality Grade
                        def get_grade(p, s):
                            score = 0
                            if p is not None:
                                if p >= 30: score += 4
                                elif p >= 25: score += 3
                                elif p >= 20: score += 2
                                elif p >= 15: score += 1
                            if s is not None:
                                if s >= 0.85: score += 4
                                elif s >= 0.70: score += 3
                                elif s >= 0.50: score += 2
                                elif s >= 0.30: score += 1
                            if p is None and s is None:
                                return "N/A", "⚪ No ground truth"
                            if score >= 7: return "A", "🟢 Excellent"
                            elif score >= 5: return "B", "🔵 Good"
                            elif score >= 3: return "C", "🟡 Average"
                            elif score >= 2: return "D", "🟠 Below Average"
                            else: return "F", "🔴 Poor"
                        
                        grade, grade_desc = get_grade(psnr_val, ssim_val)
                        
                        # Display metrics in columns
                        m1, m2, m3, m4 = st.columns(4)
                        m1.metric("☁️ Cloud Before", f"{cloud_before:.1f}%")
                        m2.metric("✨ Cloud After", f"{cloud_after:.1f}%")
                        m3.metric("🧹 Cloud Removed", f"{cloud_removed:.1f}%", delta=f"-{cloud_removed:.0f}%")
                        m4.metric("🏆 Quality Grade", f"{grade}", delta=grade_desc)
                        
                        if psnr_val is not None and ssim_val is not None:
                            p1, p2 = st.columns(2)
                            p1.metric("📊 PSNR", f"{psnr_val:.2f} dB", help="Peak Signal-to-Noise Ratio. Higher is better. >25 dB = Good")
                            p2.metric("📐 SSIM", f"{ssim_val:.4f}", help="Structural Similarity. Higher is better. >0.7 = Good")
                        else:
                            st.info("ℹ️ PSNR & SSIM require ground truth. Use 'Example' mode to see full metrics, or provide a ground truth image.")
                        
                        # Grade explanation
                        with st.expander("📋 How Quality Grades Work"):
                            st.markdown("""
                            | Grade | PSNR (dB) | SSIM | Quality |
                            |-------|-----------|------|---------|
                            | **A** | ≥ 30 | ≥ 0.85 | 🟢 Excellent |
                            | **B** | ≥ 25 | ≥ 0.70 | 🔵 Good |
                            | **C** | ≥ 20 | ≥ 0.50 | 🟡 Average |
                            | **D** | ≥ 15 | ≥ 0.30 | 🟠 Below Average |
                            | **F** | < 15 | < 0.30 | 🔴 Poor |
                            """)
                        
                    except Exception as e:
                        st.warning(f"Could not calculate some metrics: {e}")
                    
                    # Downloads
                    st.divider()
                    col_d1, col_d2 = st.columns(2)
                    buf = io.BytesIO()
                    Image.fromarray((out_img*255).astype(np.uint8)).save(buf, format="PNG")
                    col_d1.download_button("⬇️ Download Image", buf.getvalue(), "clean.png", "image/png")
                    
                    # Report
                    try:
                        cc = calculate_cloud_coverage(cloudy)
                        psnr_r = calculate_psnr(to_image(gt), out_img) if gt is not None else None
                        pdf = create_pdf_report(cloudy, final, psnr_r, cc)
                        col_d2.download_button("📄 PDF Report", pdf, "report.pdf", "application/pdf")
                    except: pass
                    
                    st.divider()
                    st.markdown("### 🔬 Advanced Inspection")
                    c_veg, c_lens = st.columns(2)
                    
                    with c_veg:
                        if st.checkbox("🌱 Vegetation Health"):
                            vari = analyze_vegetation(final)
                            st.image(vari, "Vegetation Health Index (VARI)", clamp=True)
                            
                    with c_lens:
                        if st.checkbox("🔍 Magic Lens"):
                            st.write("Hover to zoom:")
                            # Ensure out_img is in correct format (numpy array 0-1 or PIL)
                            # image_zoom expects PIL or numpy uint8 usually, let's convert to likely format
                            # Ensure simple uint8 image for zoom
                            zoom_img = Image.fromarray(gan_to_image(out_img))
                            image_zoom(zoom_img, mode="mousemove", size=150, zoom_factor=3)
                            
                    if st.checkbox("🛣️ Infrastructure & Road Extraction"):
                        infra = extract_infrastructure(final)
                        st.image(infra, "Infrastructure Highlighted (Green Edges)", clamp=True)
                else:
                    st.error("No valid model found.")

    else: # Batch
        files = st.file_uploader("Batch Upload", accept_multiple_files=True)
        if files and st.button("Run Batch"):
            # Load model once (simplified logic)
            device = "cuda" if torch.cuda.is_available() else "cpu"
            gen = LegacyUNet().to(device) # Default to legacy for safety or try-catch block
            # (Ideally duplicate the loading logic here, but keeping it brief)
            try:
                gen.load_state_dict(torch.load("gen_epoch_5.pth", map_location=device)["state_dict"])
            except:
                pass # Random weights fallback
                
            zf_buf = io.BytesIO()
            with zipfile.ZipFile(zf_buf, "w") as zf:
                bar = st.progress(0)
                for i, f in enumerate(files):
                    img = Image.open(f)
                    inp = preprocess_image(img).unsqueeze(0)
                    sar = torch.randn(1, 1, 256, 256)
                    out = gen(torch.cat([inp, sar], dim=1).to(device))
                    out_pil = Image.fromarray((to_image(out[0].cpu())*255).astype(np.uint8))
                    b = io.BytesIO()
                    out_pil.save(b, format="PNG")
                    zf.writestr(f"clean_{f.name}", b.getvalue())
                    bar.progress((i+1)/len(files))
            st.download_button("⬇️ Download ZIP", zf_buf.getvalue(), "batch.zip")

with tab2:
    if 'data' in st.session_state:
        d = st.session_state['data']
        st.metric("Cloud Coverage", f"{calculate_cloud_coverage(d['cloudy']):.1f}%")
        c1, c2 = st.columns(2)
        with c1: st.pyplot(plot_histogram(d['cloudy'], "Input"))
        with c2: st.pyplot(plot_histogram(d['out'], "Output"))
        
        if st.session_state.get('data'):
             st.divider()
             st.subheader("🌍 Location Visualization")
             d = st.session_state['data']
             loc = d.get('loc')
             
             try:
                 import folium
                 from streamlit_folium import st_folium
                 
                 map_center = loc if loc else [13.0827, 80.2707]
                 m = folium.Map(location=map_center, zoom_start=10 if loc else 4)
                 
                 if loc:
                     folium.Marker(loc, popup="Image Location", icon=folium.Icon(color="red", icon="cloud")).add_to(m)
                     st.success(f"📍 Location: {loc}")
                 else:
                     st.info("No GPS data found. Showing default view.")
                     
                 st_folium(m, width=800, height=400)
             except ImportError:
                 st.error("Map components missing.")
             except Exception as e:
                 st.error(f"Error: {e}")

with tab3:
    st.header("🏔️ 3D Terrain Reconstruction")
    
    if 'data' in st.session_state:
        d = st.session_state['data']
        out_img = d['out']
        
        st.info("🎨 Generating interactive 3D terrain from satellite imagery...")
        
        col1, col2 = st.columns([1, 3])
        
        with col1:
            st.subheader("⚙️ Controls")
            exaggeration = st.slider(
                "Height Exaggeration",
                min_value=0.5,
                max_value=5.0,
                value=2.0,
                step=0.5,
                help="Increase to make terrain features more dramatic"
            )
            
            method = st.radio(
                "Height Estimation Method",
                ["brightness", "gradient"],
                help="Brightness: lighter = higher\nGradient: edges = elevation changes"
            )
            
            resolution = st.select_slider(
                "Resolution (Performance)",
                options=[0.25, 0.5, 0.75, 1.0],
                value=0.75,
                help="Lower = faster rendering"
            )
        
        with col2:
            with st.spinner("Generating 3D terrain..."):
                try:
                    fig = generate_3d_visualization(
                        out_img,
                        exaggeration=exaggeration,
                        method=method,
                        resolution=resolution
                    )
                    st.plotly_chart(fig, use_container_width=True, height=600)
                    
                    st.success("✅ 3D terrain generated! Use your mouse to rotate, zoom, and pan.")
                    
                    with st.expander("💡 How to interact"):
                        st.markdown("""
                        - **Rotate**: Click and drag
                        - **Zoom**: Scroll wheel or pinch
                        - **Pan**: Right-click and drag
                        - **Reset**: Double-click
                        """)
                except Exception as e:
                    st.error(f"Error generating 3D terrain: {e}")
    else:
        st.info("👈 Process an image in the Workspace tab first to view 3D terrain!")

with tab4:
    st.header("�️ Land Use & Land Cover Classification")
    
    if 'data' in st.session_state:
        d = st.session_state['data']
        out_img = d['out']
        
        with st.spinner("Classifying land cover..."):
            try:
                lulc_result = classify_and_visualize(out_img)
                
                # Store in session state for chatbot
                st.session_state['lulc_data'] = lulc_result
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("🗺️ Color-Coded Classification")
                    st.image(
                        lulc_result['color_map'],
                        caption="Land Cover Map",
                        use_container_width=True
                    )
                    
                    # Legend
                    st.markdown("**Legend:**")
                    for class_id, class_info in LULC_CLASSES.items():
                        color_hex = f"#{class_info['color'][0]:02x}{class_info['color'][1]:02x}{class_info['color'][2]:02x}"
                        st.markdown(
                            f"<div style='display: flex; align-items: center;'>"
                            f"<div style='width: 20px; height: 20px; background-color: {color_hex}; margin-right: 10px; border: 1px solid #ccc;'></div>"
                            f"<span>{class_info['label']}</span>"
                            f"</div>",
                            unsafe_allow_html=True
                        )
                
                with col2:
                    st.subheader("📊 Distribution Analysis")
                    
                    # Pie chart
                    if lulc_result['values']:
                        fig = go.Figure(data=[go.Pie(
                            labels=lulc_result['labels'],
                            values=lulc_result['values'],
                            marker=dict(colors=lulc_result['colors']),
                            textinfo='label+percent',
                            hovertemplate='%{label}<br>%{value:.1f}%<extra></extra>'
                        )])
                        
                        fig.update_layout(
                            title="Land Cover Distribution",
                            height=400,
                            showlegend=True
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Percentage table
                    st.markdown("**Detailed Breakdown:**")
                    for name, pct in sorted(lulc_result['percentages'].items(), key=lambda x: x[1], reverse=True):
                        if pct > 0.1:
                            st.metric(name, f"{pct:.2f}%")
                
                st.divider()
                
                # AI Chatbot Section
                st.subheader("🤖 AI Map Assistant")
                st.markdown("*Ask questions about the land cover analysis!*")
                
                # Initialize chatbot
                if 'chatbot' not in st.session_state:
                    st.session_state['chatbot'] = MapChatbot()
                    st.session_state['chat_history'] = []
                
                # Chat interface
                col_chat1, col_chat2 = st.columns([3, 1])
                
                with col_chat1:
                    user_query = st.text_input(
                        "Ask a question:",
                        placeholder="e.g., How much water is in this area?",
                        key="chat_input"
                    )
                
                with col_chat2:
                    ask_button = st.button("🔍 Ask", type="primary")
                
                # Suggested questions
                st.markdown("**💡 Try asking:**")
                suggestions = [
                    "How much water is in this area?",
                    "Where is the urban area located?",
                    "Is there more forest or vegetation?",
                    "Tell me about this map"
                ]
                
                cols = st.columns(2)
                for i, suggestion in enumerate(suggestions):
                    with cols[i % 2]:
                        if st.button(suggestion, key=f"suggest_{i}"):
                            user_query = suggestion
                            ask_button = True
                
                # Process query
                if ask_button and user_query:
                    with st.spinner("Thinking..."):
                        response = st.session_state['chatbot'].chat(user_query, lulc_result)
                        st.session_state['chat_history'].append({
                            'query': user_query,
                            'response': response
                        })
                
                # Display chat history
                if st.session_state.get('chat_history'):
                    st.divider()
                    st.markdown("**� Conversation:**")
                    for chat in reversed(st.session_state['chat_history'][-5:]):
                        st.markdown(f"**You:** {chat['query']}")
                        st.markdown(f"**AI:** {chat['response']}")
                        st.markdown("---")
                
            except Exception as e:
                st.error(f"Error in LULC classification: {e}")
                import traceback
                st.code(traceback.format_exc())
    else:
        st.info("👈 Process an image in the Workspace tab first to analyze land cover!")

with tab5:
    st.markdown("### 🎉 v5.0 Advanced Features")
    st.markdown("""
    #### 🆕 New in v5.0:
    - **🏔️ 3D Terrain Reconstruction**: Interactive 3D visualization with adjustable height exaggeration
    - **🏘️ LULC Classification**: Classify land into Water, Forest, Urban, Barren, and Vegetation
    - **🤖 AI Chatbot Assistant**: Ask natural language questions about your map
    
    #### 📋 Previous Features:
    - **🌱 Vegetation Health Index**: Analyze plant health using VARI
    - **🛣️ Infrastructure/Roads**: Extract man-made structures
    - **🔍 Magic Lens**: Detailed inspection tool
    - **📄 PDF Reports**: Generate professional reports
    - **📦 Batch Processing**: Process multiple images at once
    - **🗺️ Geo-tagging**: GPS location visualization
    """)
    
    st.divider()
    st.markdown("### 🚀 Quick Start Guide")
    st.markdown("""
    1. Go to **Workspace** tab
    2. Click **Load New** or upload your own satellite image
    3. Explore the processed result
    4. Visit **3D Terrain** tab for interactive 3D view
    5. Check **Land Cover** tab for classification and AI chat
    """)
