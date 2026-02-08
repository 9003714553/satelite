import torch

# Load the checkpoint
try:
    checkpoint = torch.load("gen_epoch_5.pth", map_location="cpu")
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    print("Keys in checkpoint:")
    for key, val in state_dict.items():
        print(f"{key}: {val.shape}")
        
except Exception as e:
    print(f"Error loading checkpoint: {e}")
