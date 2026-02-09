"""
Quick Start Script for Training Cloud Removal Model
Simplified interface for beginners
"""

import os
import sys
import subprocess

def print_header(text):
    print("\n" + "=" * 80)
    print(f"  {text}")
    print("=" * 80)

def check_dataset():
    """Check if dataset exists and is properly structured"""
    data_dir = "./data"
    required_folders = ["cloudy", "clear"]
    
    if not os.path.exists(data_dir):
        return False, "Data directory not found"
    
    for folder in required_folders:
        folder_path = os.path.join(data_dir, folder)
        if not os.path.exists(folder_path):
            return False, f"Missing folder: {folder}"
        
        files = os.listdir(folder_path)
        if len(files) == 0:
            return False, f"Empty folder: {folder}"
    
    # Count images
    cloudy_count = len(os.listdir(os.path.join(data_dir, "cloudy")))
    clear_count = len(os.listdir(os.path.join(data_dir, "clear")))
    
    if cloudy_count != clear_count:
        return False, f"Mismatch: {cloudy_count} cloudy images vs {clear_count} clear images"
    
    return True, f"Found {cloudy_count} image pairs"

def main():
    print_header("🛰️  CLOUD REMOVAL MODEL - QUICK START TRAINING")
    
    print("\n📋 Pre-flight Checklist:")
    print("   [1/3] Checking Python environment...")
    
    try:
        import torch
        print(f"   ✅ PyTorch {torch.__version__} installed")
        if torch.cuda.is_available():
            print(f"   ✅ GPU available: {torch.cuda.get_device_name(0)}")
        else:
            print("   ⚠️  No GPU found - training will be slow on CPU")
    except ImportError:
        print("   ❌ PyTorch not installed!")
        print("      Install with: pip install torch torchvision")
        return
    
    print("\n   [2/3] Checking dataset...")
    dataset_ok, message = check_dataset()
    
    if dataset_ok:
        print(f"   ✅ Dataset ready: {message}")
        use_real_data = True
    else:
        print(f"   ⚠️  Dataset issue: {message}")
        print("   📝 Will use mock data for testing")
        use_real_data = False
        
        response = input("\n   Continue with mock data? (y/n): ").lower()
        if response != 'y':
            print("\n   Please prepare your dataset first:")
            print("   1. Create folders: data/cloudy and data/clear")
            print("   2. Add matching satellite images to both folders")
            print("   3. Run this script again")
            return
    
    print("\n   [3/3] Checking training script...")
    if os.path.exists("src/train_cloud_removal.py"):
        print("   ✅ Training script found")
    else:
        print("   ❌ Training script not found!")
        return
    
    # Get training parameters
    print_header("⚙️  TRAINING CONFIGURATION")
    
    print("\n🎯 Recommended settings:")
    print("   • Epochs: 20 (good balance)")
    print("   • Batch size: 4 (works on most GPUs)")
    print("   • Learning rate: 0.0002 (standard)")
    
    use_defaults = input("\n   Use recommended settings? (y/n): ").lower()
    
    if use_defaults == 'y':
        epochs = 20
        batch_size = 4
        learning_rate = 0.0002
    else:
        try:
            epochs = int(input("   Epochs (10-50): ") or "20")
            batch_size = int(input("   Batch size (1-8): ") or "4")
            learning_rate = float(input("   Learning rate (0.0001-0.001): ") or "0.0002")
        except ValueError:
            print("   Invalid input! Using defaults.")
            epochs = 20
            batch_size = 4
            learning_rate = 0.0002
    
    # Build command
    cmd = [
        sys.executable,  # Python executable
        "src/train_cloud_removal.py",
        "--epochs", str(epochs),
        "--batch_size", str(batch_size),
        "--learning_rate", str(learning_rate),
        "--checkpoint_dir", "./checkpoints",
        "--save_every", "5"
    ]
    
    if not use_real_data:
        cmd.append("--use_mock_data")
    
    # Confirm and start
    print_header("🚀 READY TO START TRAINING")
    
    print(f"\n📊 Training Summary:")
    print(f"   • Epochs: {epochs}")
    print(f"   • Batch size: {batch_size}")
    print(f"   • Learning rate: {learning_rate}")
    print(f"   • Data: {'Real dataset' if use_real_data else 'Mock data (testing)'}")
    print(f"   • Checkpoints: ./checkpoints/")
    
    if torch.cuda.is_available():
        estimated_time = epochs * 1  # ~1 minute per epoch on GPU
        print(f"   • Estimated time: ~{estimated_time} minutes (GPU)")
    else:
        estimated_time = epochs * 5  # ~5 minutes per epoch on CPU
        print(f"   • Estimated time: ~{estimated_time} minutes (CPU)")
    
    print("\n⚠️  Training will start in 5 seconds...")
    print("   Press Ctrl+C to cancel")
    
    try:
        import time
        for i in range(5, 0, -1):
            print(f"   {i}...", end='\r')
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n\n   ❌ Training cancelled by user")
        return
    
    print("\n")
    print_header("🎓 TRAINING STARTED")
    print("\n💡 Tips:")
    print("   • Watch the Generator Loss - it should decrease")
    print("   • Checkpoints save every 5 epochs")
    print("   • Press Ctrl+C to stop training early")
    print("\n" + "=" * 80 + "\n")
    
    # Run training
    try:
        subprocess.run(cmd, check=True)
        
        print_header("✅ TRAINING COMPLETED SUCCESSFULLY!")
        print("\n📦 Next Steps:")
        print("   1. Find your trained models in: ./checkpoints/")
        print("   2. Copy best checkpoint: copy checkpoints\\gen_epoch_20.pth gen_epoch_5.pth")
        print("   3. Test the model: python -m streamlit run src/app.py")
        print("\n🎉 Happy testing!")
        
    except subprocess.CalledProcessError:
        print_header("❌ TRAINING FAILED")
        print("\n🔧 Troubleshooting:")
        print("   • Check error messages above")
        print("   • Reduce batch size if out of memory")
        print("   • Verify dataset is correct")
        
    except KeyboardInterrupt:
        print("\n\n" + "=" * 80)
        print("  ⚠️  TRAINING INTERRUPTED BY USER")
        print("=" * 80)
        print("\n💾 Partial checkpoints may have been saved in ./checkpoints/")
        print("   You can resume training later with --pretrained_checkpoint")

if __name__ == "__main__":
    main()
