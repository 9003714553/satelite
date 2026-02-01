import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from models import UNetGenerator
from data_loader import get_dataloader

# Configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT_FILE = "gen_epoch_10.pth" # Example checkpoint

def load_checkpoint(checkpoint_file, model):
    print(f"=> Loading checkpoint {checkpoint_file}")
    checkpoint = torch.load(checkpoint_file, map_location=DEVICE)
    model.load_state_dict(checkpoint["state_dict"])

def visualize_results(cloudy, sar, generated, clear):
    # Move to CPU and numpy
    cloudy = cloudy.permute(1, 2, 0).cpu().numpy()
    sar = sar.permute(1, 2, 0).cpu().numpy()
    generated = generated.permute(1, 2, 0).cpu().numpy()
    clear = clear.permute(1, 2, 0).cpu().numpy()
    
    # Normalize for display if needed (assuming tanh output -1 to 1)
    generated = (generated + 1) / 2
    clear = (clear + 1) / 2
    cloudy = (cloudy + 1) / 2 # Assuming input was also normalized
    
    fig, ax = plt.subplots(1, 4, figsize=(20, 5))
    ax[0].imshow(cloudy)
    ax[0].set_title("Input Cloudy")
    ax[1].imshow(sar, cmap='gray')
    ax[1].set_title("Input SAR")
    ax[2].imshow(generated)
    ax[2].set_title("Generated Clear")
    ax[3].imshow(clear)
    ax[3].set_title("Ground Truth")
    plt.show()

def run_inference():
    gen = UNetGenerator().to(DEVICE)
    
    # In a real scenario, you would load a trained model
    # if os.path.exists(CHECKPOINT_FILE):
    #     load_checkpoint(CHECKPOINT_FILE, gen)
    # else:
    #     print("Checkpoint not found, visualizing with random weights...")

    gen.eval()
    
    loader = get_dataloader(batch_size=1, split='val', mock_data=True)
    
    with torch.no_grad():
        batch = next(iter(loader))
        cloudy = batch['cloudy'].to(DEVICE)
        sar = batch['sar'].to(DEVICE)
        clear = batch['clear'].to(DEVICE)
        
        gen_input = torch.cat([cloudy, sar], dim=1)
        fake_clear = gen(gen_input)
        
        visualize_results(cloudy[0], sar[0], fake_clear[0], clear[0])

if __name__ == "__main__":
    run_inference()
