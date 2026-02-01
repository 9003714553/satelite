import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import ChangeDetectionDataset
from models import SiameseNetwork

def train(args):
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Dataset and DataLoader
    # If no data is present, the dataset class will generate dummy data for testing
    train_dataset = ChangeDetectionDataset(root_dir=args.data_dir, mode='train')
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    # Model
    model = SiameseNetwork().to(device)

    # Loss and Optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    print("Starting training...")
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        
        # Wrap loader with tqdm for progress bar
        pbar = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{args.epochs}]")
        for i, (img1, img2, mask) in enumerate(pbar):
            img1 = img1.to(device)
            img2 = img2.to(device)
            mask = mask.to(device)

            # Forward pass
            outputs = model(img1, img2)
            loss = criterion(outputs, mask)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
        
        epoch_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{args.epochs}], Average Loss: {epoch_loss:.4f}")
        
        # Save checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            os.makedirs(args.checkpoint_dir, exist_ok=True)
            ckpt_path = os.path.join(args.checkpoint_dir, f'model_epoch_{epoch+1}.pth')
            torch.save(model.state_dict(), ckpt_path)
            print(f"Model saved to {ckpt_path}")

    # Save final model
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    final_path = os.path.join(args.checkpoint_dir, 'model_final.pth')
    torch.save(model.state_dict(), final_path)
    print(f"Final model saved to {final_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Change Detection Model')
    parser.add_argument('--data_dir', type=str, default='./data', help='Path to dataset')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints', help='Directory to save checkpoints')
    
    args = parser.parse_args()
    
    # Create dummy data dir if it doesn't exist just to suppress errors if user runs blindly
    os.makedirs(args.data_dir, exist_ok=True)
    
    train(args)
