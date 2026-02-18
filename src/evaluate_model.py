import sys
import io
# Fix Windows terminal encoding for Unicode/emoji support
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

"""
🛰️ Cloud Removal AI - Model Evaluation Script
================================================
Evaluates the trained GAN model using standard image quality metrics:
  - PSNR (Peak Signal-to-Noise Ratio)
  - SSIM (Structural Similarity Index)
  - Cloud Coverage Reduction %
  - Quality Grade (A/B/C/D/F)

Usage:
  python src/evaluate_model.py                    # Evaluate with mock data
  python src/evaluate_model.py --num_samples 20   # Custom number of samples
"""

import torch
import torch.nn as nn
import numpy as np
import os
import sys
from PIL import Image
import torchvision.transforms as T
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models import UNetGenerator
from data_loader import get_dataloader


# =============================================
# Legacy Model (matches old checkpoint format)
# =============================================
class LegacyUNet(nn.Module):
    def __init__(self):
        super(LegacyUNet, self).__init__()
        self.enc1 = nn.Sequential(nn.Conv2d(4, 64, 4, 2, 1, bias=False), nn.LeakyReLU(0.2))
        self.enc2 = nn.Sequential(nn.Conv2d(64, 128, 4, 2, 1, bias=False), nn.LeakyReLU(0.2))
        self.enc3 = nn.Sequential(nn.Conv2d(128, 256, 4, 2, 1, bias=False), nn.LeakyReLU(0.2))
        self.enc4 = nn.Sequential(nn.Conv2d(256, 512, 4, 2, 1, bias=False), nn.LeakyReLU(0.2))
        self.bottleneck = nn.Sequential(nn.Conv2d(512, 512, 4, 2, 1, bias=False), nn.ReLU())
        self.up4 = nn.ConvTranspose2d(512, 512, 4, 2, 1)
        self.dec4 = nn.Sequential(nn.Conv2d(1024, 256, 3, 1, 1, bias=False), nn.ReLU())
        self.up3 = nn.ConvTranspose2d(256, 256, 4, 2, 1)
        self.dec3 = nn.Sequential(nn.Conv2d(512, 128, 3, 1, 1, bias=False), nn.ReLU())
        self.up2 = nn.ConvTranspose2d(128, 128, 4, 2, 1)
        self.dec2 = nn.Sequential(nn.Conv2d(256, 64, 3, 1, 1, bias=False), nn.ReLU())
        self.up1 = nn.ConvTranspose2d(64, 64, 4, 2, 1)
        self.dec1 = nn.Sequential(nn.Conv2d(128, 64, 3, 1, 1, bias=False), nn.ReLU())
        self.final = nn.Sequential(nn.ReflectionPad2d(1), nn.Conv2d(64, 3, 3, 1, 0), nn.Tanh())

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        b = self.bottleneck(e4)
        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.final(d1)


# =============================================
# Metric Calculation Functions
# =============================================

def tensor_to_numpy(tensor):
    """Convert tensor [C, H, W] in [-1, 1] to numpy [H, W, C] in [0, 1]"""
    img = tensor.detach().permute(1, 2, 0).cpu().numpy()
    # Normalize to [0, 1] range
    img_min, img_max = img.min(), img.max()
    if img_max - img_min > 0:
        img = (img - img_min) / (img_max - img_min)
    else:
        img = np.zeros_like(img)
    return np.clip(img, 0, 1)


def _match_dimensions(gen_np, tgt_np):
    """Resize generated image to match target dimensions if needed"""
    if gen_np.shape != tgt_np.shape:
        from PIL import Image as PILImage
        # Resize gen to match target
        h, w = tgt_np.shape[:2]
        gen_pil = PILImage.fromarray((gen_np * 255).astype(np.uint8))
        gen_pil = gen_pil.resize((w, h), PILImage.BILINEAR)
        gen_np = np.array(gen_pil).astype(np.float64) / 255.0
    return gen_np, tgt_np


def calculate_psnr(generated, target):
    """Calculate PSNR between generated and target images"""
    gen_np = tensor_to_numpy(generated)
    tgt_np = tensor_to_numpy(target)
    gen_np, tgt_np = _match_dimensions(gen_np, tgt_np)
    return psnr(tgt_np, gen_np, data_range=1.0)


def calculate_ssim(generated, target):
    """Calculate SSIM between generated and target images"""
    gen_np = tensor_to_numpy(generated)
    tgt_np = tensor_to_numpy(target)
    gen_np, tgt_np = _match_dimensions(gen_np, tgt_np)
    return ssim(tgt_np, gen_np, channel_axis=2, data_range=1.0)


def calculate_cloud_coverage(img_tensor):
    """Estimate cloud coverage percentage based on brightness"""
    img = tensor_to_numpy(img_tensor)
    brightness = np.mean(img, axis=2)
    return (np.sum(brightness > 0.6) / brightness.size) * 100


def get_quality_grade(psnr_val, ssim_val):
    """Assign quality grade based on PSNR and SSIM"""
    score = 0
    
    # PSNR scoring
    if psnr_val >= 30: score += 4
    elif psnr_val >= 25: score += 3
    elif psnr_val >= 20: score += 2
    elif psnr_val >= 15: score += 1
    
    # SSIM scoring
    if ssim_val >= 0.85: score += 4
    elif ssim_val >= 0.70: score += 3
    elif ssim_val >= 0.50: score += 2
    elif ssim_val >= 0.30: score += 1
    
    # Grade mapping
    if score >= 7: return "A", "🟢 Excellent"
    elif score >= 5: return "B", "🔵 Good"
    elif score >= 3: return "C", "🟡 Average"
    elif score >= 2: return "D", "🟠 Below Average"
    else: return "F", "🔴 Poor"


# =============================================
# Model Loading
# =============================================

def load_model(device):
    """Try to load the best available model"""
    
    # Possible checkpoint paths
    checkpoint_paths = [
        "gen_epoch_10.pth",
        "gen_epoch_5.pth",
        "src/gen_epoch_10.pth",
        "src/gen_epoch_5.pth",
        "checkpoints/gen_final.pth",
    ]
    
    for chk_path in checkpoint_paths:
        if not os.path.exists(chk_path):
            continue
        
        print(f"  Found checkpoint: {chk_path}")
        checkpoint = torch.load(chk_path, map_location=device, weights_only=False)
        sd = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint
        
        # Try Standard Model first
        try:
            model = UNetGenerator(input_channels=4, output_channels=3).to(device)
            model.load_state_dict(sd)
            print(f"  ✅ Loaded Standard UNet Model")
            return model
        except Exception:
            pass
        
        # Try Legacy Model
        try:
            model = LegacyUNet().to(device)
            model.load_state_dict(sd)
            print(f"  ✅ Loaded Legacy UNet Model")
            return model
        except Exception:
            pass
    
    print("  ⚠️  No checkpoint found. Using untrained model (random weights).")
    model = LegacyUNet().to(device)
    return model


# =============================================
# Main Evaluation
# =============================================

def evaluate(num_samples=10, save_results=False):
    """Run full model evaluation"""
    
    print("=" * 70)
    print("🛰️  CLOUD REMOVAL AI - MODEL EVALUATION")
    print("=" * 70)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n📌 Device: {device}")
    
    # Load model
    print(f"\n📦 Loading model...")
    gen = load_model(device)
    gen.eval()
    
    # Load test data
    print(f"\n📊 Loading test data ({num_samples} samples)...")
    loader = get_dataloader(batch_size=1, split='val', mock_data=True)
    
    # Metrics storage
    psnr_scores = []
    ssim_scores = []
    cloud_before_list = []
    cloud_after_list = []
    
    print(f"\n🔄 Processing images...\n")
    print(f"{'#':<4} {'PSNR (dB)':<12} {'SSIM':<10} {'Cloud Before':<15} {'Cloud After':<15} {'Removed':<10}")
    print("-" * 70)
    
    with torch.no_grad():
        for i in range(num_samples):
            try:
                batch = next(iter(loader))
                cloudy = batch['cloudy'][0].to(device)
                sar = batch['sar'][0].to(device)
                clear = batch['clear'][0].to(device)
                
                # Generate clean image
                inp = torch.cat([cloudy.unsqueeze(0), sar.unsqueeze(0)], dim=1)
                fake_clean = gen(inp)[0]
                
                # Calculate metrics
                p = calculate_psnr(fake_clean, clear)
                s = calculate_ssim(fake_clean, clear)
                cloud_before = calculate_cloud_coverage(cloudy)
                cloud_after = calculate_cloud_coverage(fake_clean)
                cloud_removed = max(0, cloud_before - cloud_after)
                
                psnr_scores.append(p)
                ssim_scores.append(s)
                cloud_before_list.append(cloud_before)
                cloud_after_list.append(cloud_after)
                
                print(f"{i+1:<4} {p:<12.2f} {s:<10.4f} {cloud_before:<15.1f}% {cloud_after:<14.1f}% {cloud_removed:<10.1f}%")
                
            except Exception as e:
                print(f"{i+1:<4} Error: {e}")
    
    # =============================================
    # Summary Report
    # =============================================
    if len(psnr_scores) == 0:
        print("\n❌ No images were evaluated!")
        return
    
    avg_psnr = np.mean(psnr_scores)
    avg_ssim = np.mean(ssim_scores)
    avg_cloud_before = np.mean(cloud_before_list)
    avg_cloud_after = np.mean(cloud_after_list)
    avg_cloud_removed = max(0, avg_cloud_before - avg_cloud_after)
    grade, grade_desc = get_quality_grade(avg_psnr, avg_ssim)
    
    print("\n" + "=" * 70)
    print("📈 EVALUATION SUMMARY")
    print("=" * 70)
    print(f"""
  Samples Evaluated : {len(psnr_scores)}
  
  📊 Average Metrics:
  ┌─────────────────────────────────────────┐
  │  PSNR          : {avg_psnr:.2f} dB               │
  │  SSIM          : {avg_ssim:.4f}                 │
  │  Cloud Before  : {avg_cloud_before:.1f}%                  │
  │  Cloud After   : {avg_cloud_after:.1f}%                  │
  │  Cloud Removed : {avg_cloud_removed:.1f}%                  │
  └─────────────────────────────────────────┘
  
  🏆 Quality Grade: {grade} - {grade_desc}
  
  📋 Grade Scale:
     A (🟢 Excellent) : PSNR ≥ 30, SSIM ≥ 0.85
     B (🔵 Good)      : PSNR ≥ 25, SSIM ≥ 0.70
     C (🟡 Average)   : PSNR ≥ 20, SSIM ≥ 0.50
     D (🟠 Below Avg) : PSNR ≥ 15, SSIM ≥ 0.30
     F (🔴 Poor)      : PSNR < 15, SSIM < 0.30
""")
    
    # Best and worst
    best_idx = np.argmax(psnr_scores)
    worst_idx = np.argmin(psnr_scores)
    print(f"  Best Sample  : #{best_idx+1} (PSNR={psnr_scores[best_idx]:.2f}, SSIM={ssim_scores[best_idx]:.4f})")
    print(f"  Worst Sample : #{worst_idx+1} (PSNR={psnr_scores[worst_idx]:.2f}, SSIM={ssim_scores[worst_idx]:.4f})")
    print("=" * 70)
    
    return {
        'avg_psnr': avg_psnr,
        'avg_ssim': avg_ssim,
        'avg_cloud_removed': avg_cloud_removed,
        'grade': grade,
        'grade_desc': grade_desc,
        'psnr_scores': psnr_scores,
        'ssim_scores': ssim_scores,
    }


# =============================================
# Evaluate Single Image
# =============================================

def evaluate_single_image(cloudy_path, sar_path=None, ground_truth_path=None):
    """Evaluate model on a single user-provided image"""
    
    print("=" * 70)
    print("🛰️  SINGLE IMAGE EVALUATION")
    print("=" * 70)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load and preprocess
    transform_rgb = T.Compose([T.Resize((256, 256)), T.ToTensor(), T.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))])
    transform_sar = T.Compose([T.Resize((256, 256)), T.ToTensor(), T.Normalize((0.5,), (0.5,))])
    
    # Load cloudy image
    cloudy_img = Image.open(cloudy_path).convert('RGB')
    cloudy = transform_rgb(cloudy_img)
    
    # Load or generate SAR
    if sar_path and os.path.exists(sar_path):
        sar_img = Image.open(sar_path).convert('L')
        sar = transform_sar(sar_img)
    else:
        print("  ⚠️  No SAR image provided. Using random noise as fallback.")
        sar = torch.randn(1, 256, 256)
    
    # Load model
    gen = load_model(device)
    gen.eval()
    
    # Generate
    with torch.no_grad():
        inp = torch.cat([cloudy.unsqueeze(0), sar.unsqueeze(0)], dim=1).to(device)
        fake_clean = gen(inp)[0].cpu()
    
    # Metrics
    cloud_before = calculate_cloud_coverage(cloudy)
    cloud_after = calculate_cloud_coverage(fake_clean)
    
    print(f"\n  📊 Results:")
    print(f"     Cloud Before : {cloud_before:.1f}%")
    print(f"     Cloud After  : {cloud_after:.1f}%")
    print(f"     Cloud Removed: {max(0, cloud_before - cloud_after):.1f}%")
    
    if ground_truth_path and os.path.exists(ground_truth_path):
        gt_img = Image.open(ground_truth_path).convert('RGB')
        gt = transform_rgb(gt_img)
        p = calculate_psnr(fake_clean, gt)
        s = calculate_ssim(fake_clean, gt)
        grade, grade_desc = get_quality_grade(p, s)
        print(f"     PSNR         : {p:.2f} dB")
        print(f"     SSIM         : {s:.4f}")
        print(f"     Grade        : {grade} - {grade_desc}")
    else:
        print(f"     ℹ️  No ground truth provided. PSNR/SSIM not calculated.")
        print(f"     ℹ️  (Without a clear reference image, only cloud coverage can be measured)")
    
    print("=" * 70)
    return fake_clean


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate Cloud Removal Model")
    parser.add_argument("--num_samples", type=int, default=10, help="Number of test samples")
    parser.add_argument("--image", type=str, default=None, help="Path to a single cloudy image")
    parser.add_argument("--sar", type=str, default=None, help="Path to SAR image (optional)")
    parser.add_argument("--ground_truth", type=str, default=None, help="Path to ground truth clean image")
    
    args = parser.parse_args()
    
    if args.image:
        evaluate_single_image(args.image, args.sar, args.ground_truth)
    else:
        evaluate(num_samples=args.num_samples)
