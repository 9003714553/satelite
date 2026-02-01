import torch
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

def calculate_metrics(generated_img, target_img):
    """
    Calculate SSIM and PSNR for a batch of images.
    Inputs: Tensor [C, H, W] in range [-1, 1]
    """
    # Convert to numpy and range [0, 1] for metrics
    gen_np = (generated_img.permute(1, 2, 0).cpu().numpy() + 1) / 2
    tgt_np = (target_img.permute(1, 2, 0).cpu().numpy() + 1) / 2
    
    # Ensure range [0, 1]
    gen_np = np.clip(gen_np, 0, 1)
    tgt_np = np.clip(tgt_np, 0, 1)
    
    val_ssim = ssim(tgt_np, gen_np, channel_axis=2, data_range=1.0)
    val_psnr = psnr(tgt_np, gen_np, data_range=1.0)
    
    return val_ssim, val_psnr

def calculate_fid(real_features, fake_features):
    """
    Calculate Fréchet Inception Distance (FID).
    This assumes you have extracted features (e.g. from InceptionV3) as numpy arrays.
    real_features: [N, feature_dim]
    fake_features: [N, feature_dim]
    """
    import scipy.linalg
    
    mu1, sigma1 = real_features.mean(axis=0), np.cov(real_features, rowvar=False)
    mu2, sigma2 = fake_features.mean(axis=0), np.cov(fake_features, rowvar=False)
    
    ssdiff = np.sum((mu1 - mu2)**2)
    
    # Calculate sqrt of product of covariances
    covmean = scipy.linalg.sqrtm(sigma1.dot(sigma2))
    
    # Check and correction for imaginary numbers from sqrt
    if np.iscomplexobj(covmean):
        covmean = covmean.real
        
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid

if __name__ == "__main__":
    # Test SSIM/PSNR
    img1 = torch.randn(3, 256, 256)
    img2 = torch.randn(3, 256, 256)
    
    s, p = calculate_metrics(img1, img2)
    print(f"SSIM: {s:.4f}")
    print(f"PSNR: {p:.4f}")
    
    # Test FID with random features
    feat1 = np.random.randn(100, 2048) # 100 images, 2048 features
    feat2 = np.random.randn(100, 2048)
    fid = calculate_fid(feat1, feat2)
    print(f"FID: {fid:.4f}")
