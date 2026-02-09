"""
Cloud Removal Model Training Script
Train GAN-based UNet model with custom satellite imagery dataset
"""

import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import time
from datetime import datetime

# Import model and dataset
from models import UNetGenerator
from data_loader import CloudRemovalDataset


class Discriminator(nn.Module):
    """PatchGAN Discriminator for realistic image generation"""
    def __init__(self, input_channels=3):
        super(Discriminator, self).__init__()
        
        def discriminator_block(in_filters, out_filters, normalization=True):
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalization:
                layers.append(nn.BatchNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers
        
        self.model = nn.Sequential(
            *discriminator_block(input_channels, 64, normalization=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.Conv2d(512, 1, 4, padding=1)
        )
    
    def forward(self, img):
        return self.model(img)


def train(args):
    """Main training function"""
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Create checkpoint directory
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # Dataset and DataLoader
    print(f"\nLoading dataset from: {args.data_dir}")
    train_dataset = CloudRemovalDataset(
        root_dir=args.data_dir, 
        split='train', 
        mock_data=args.use_mock_data
    )
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
        num_workers=args.num_workers
    )
    print(f"   Dataset size: {len(train_dataset)} images")
    print(f"   Batch size: {args.batch_size}")
    print(f"   Batches per epoch: {len(train_loader)}")
    
    # Models
    print(f"\nInitializing models...")
    generator = UNetGenerator(input_channels=4, output_channels=3).to(device)
    discriminator = Discriminator(input_channels=3).to(device)
    
    # Load pretrained checkpoint if specified
    start_epoch = 0
    if args.pretrained_checkpoint and os.path.exists(args.pretrained_checkpoint):
        print(f"   Loading pretrained checkpoint: {args.pretrained_checkpoint}")
        checkpoint = torch.load(args.pretrained_checkpoint, map_location=device)
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            generator.load_state_dict(checkpoint['state_dict'])
            start_epoch = checkpoint.get('epoch', 0)
        else:
            generator.load_state_dict(checkpoint)
        print(f"   Resuming from epoch {start_epoch}")
    
    # Loss functions
    criterion_GAN = nn.MSELoss()  # LSGAN loss
    criterion_pixelwise = nn.L1Loss()
    
    # Optimizers
    optimizer_G = optim.Adam(generator.parameters(), lr=args.learning_rate, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=args.learning_rate, betas=(0.5, 0.999))
    
    # Training loop
    print(f"\nStarting training for {args.epochs} epochs...")
    print(f"   Learning rate: {args.learning_rate}")
    print(f"   Lambda (pixel loss weight): {args.lambda_pixel}")
    print("-" * 80)
    
    for epoch in range(start_epoch, args.epochs):
        epoch_start_time = time.time()
        generator.train()
        discriminator.train()
        
        running_loss_G = 0.0
        running_loss_D = 0.0
        running_loss_pixel = 0.0
        
        # Progress bar
        pbar = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{args.epochs}]")
        
        for i, batch in enumerate(pbar):
            cloudy = batch['cloudy'].to(device)
            sar = batch['sar'].to(device)
            clear = batch['clear'].to(device)
            
            batch_size = cloudy.size(0)
            
            # -----------------
            #  Train Generator
            # -----------------
            optimizer_G.zero_grad()
            
            # Generate fake images
            input_combined = torch.cat([cloudy, sar], dim=1)
            fake_clear = generator(input_combined)
            
            # GAN loss - get actual discriminator output size
            pred_fake = discriminator(fake_clear)
            # Create target tensors with same size as discriminator output
            valid = torch.ones_like(pred_fake, device=device)
            fake = torch.zeros_like(pred_fake, device=device)
            
            loss_GAN = criterion_GAN(pred_fake, valid)
            
            # Pixel-wise loss
            loss_pixel = criterion_pixelwise(fake_clear, clear)
            
            # Total generator loss
            loss_G = loss_GAN + args.lambda_pixel * loss_pixel
            
            loss_G.backward()
            optimizer_G.step()
            
            # ---------------------
            #  Train Discriminator
            # ---------------------
            optimizer_D.zero_grad()
            
            # Real loss
            pred_real = discriminator(clear)
            loss_real = criterion_GAN(pred_real, valid)
            
            # Fake loss
            pred_fake = discriminator(fake_clear.detach())
            loss_fake = criterion_GAN(pred_fake, fake)
            
            # Total discriminator loss
            loss_D = 0.5 * (loss_real + loss_fake)
            
            loss_D.backward()
            optimizer_D.step()
            
            # Update running losses
            running_loss_G += loss_G.item()
            running_loss_D += loss_D.item()
            running_loss_pixel += loss_pixel.item()
            
            # Update progress bar
            pbar.set_postfix({
                'G_loss': f'{loss_G.item():.4f}',
                'D_loss': f'{loss_D.item():.4f}',
                'Pixel': f'{loss_pixel.item():.4f}'
            })
        
        # Epoch statistics
        epoch_time = time.time() - epoch_start_time
        avg_loss_G = running_loss_G / len(train_loader)
        avg_loss_D = running_loss_D / len(train_loader)
        avg_loss_pixel = running_loss_pixel / len(train_loader)
        
        print(f"\nEpoch [{epoch+1}/{args.epochs}] Summary:")
        print(f"   Generator Loss: {avg_loss_G:.4f}")
        print(f"   Discriminator Loss: {avg_loss_D:.4f}")
        print(f"   Pixel Loss: {avg_loss_pixel:.4f}")
        print(f"   Time: {epoch_time:.2f}s")
        
        # Save checkpoint
        if (epoch + 1) % args.save_every == 0:
            gen_path = os.path.join(args.checkpoint_dir, f'gen_epoch_{epoch+1}.pth')
            disc_path = os.path.join(args.checkpoint_dir, f'disc_epoch_{epoch+1}.pth')
            
            # Save with metadata
            torch.save({
                'epoch': epoch + 1,
                'state_dict': generator.state_dict(),
                'optimizer': optimizer_G.state_dict(),
                'loss': avg_loss_G,
            }, gen_path)
            
            torch.save({
                'epoch': epoch + 1,
                'state_dict': discriminator.state_dict(),
                'optimizer': optimizer_D.state_dict(),
                'loss': avg_loss_D,
            }, disc_path)
            
            print(f"Checkpoint saved: {gen_path}")
        
        print("-" * 80)
    
    # Save final model
    final_gen_path = os.path.join(args.checkpoint_dir, 'gen_final.pth')
    final_disc_path = os.path.join(args.checkpoint_dir, 'disc_final.pth')
    
    torch.save({
        'epoch': args.epochs,
        'state_dict': generator.state_dict(),
        'optimizer': optimizer_G.state_dict(),
    }, final_gen_path)
    
    torch.save({
        'epoch': args.epochs,
        'state_dict': discriminator.state_dict(),
        'optimizer': optimizer_D.state_dict(),
    }, final_disc_path)
    
    print(f"\nTraining completed!")
    print(f"   Final model saved: {final_gen_path}")
    print(f"   Total training time: {(time.time() - epoch_start_time) / 60:.2f} minutes")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Cloud Removal GAN Model')
    
    # Dataset parameters
    parser.add_argument('--data_dir', type=str, default='./data', 
                        help='Path to dataset directory')
    parser.add_argument('--use_mock_data', action='store_true',
                        help='Use mock data for testing (no real dataset needed)')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=20, 
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=4, 
                        help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=0.0002, 
                        help='Learning rate for optimizers')
    parser.add_argument('--lambda_pixel', type=float, default=100.0,
                        help='Weight for pixel-wise loss')
    
    # Checkpoint parameters
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints', 
                        help='Directory to save model checkpoints')
    parser.add_argument('--save_every', type=int, default=5, 
                        help='Save checkpoint every N epochs')
    parser.add_argument('--pretrained_checkpoint', type=str, default=None,
                        help='Path to pretrained checkpoint to resume from')
    
    # System parameters
    parser.add_argument('--num_workers', type=int, default=0,
                        help='Number of data loading workers (0 for Windows)')
    
    args = parser.parse_args()
    
    # Print configuration
    print("=" * 80)
    print("CLOUD REMOVAL MODEL TRAINING")
    print("=" * 80)
    print(f"Configuration:")
    for arg in vars(args):
        print(f"  {arg}: {getattr(args, arg)}")
    print("=" * 80)
    
    # Start training
    train(args)
