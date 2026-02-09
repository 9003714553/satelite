"""
Dataset Preparation Helper Script
Helps organize satellite images into proper folder structure for training
"""

import os
import shutil
from pathlib import Path
import argparse

def print_banner(text):
    print("\n" + "=" * 80)
    print(f"  {text}")
    print("=" * 80 + "\n")

def create_folder_structure(base_dir):
    """Create the required folder structure"""
    folders = ['cloudy', 'sar', 'clear']
    
    print(f"📁 Creating folder structure in: {base_dir}")
    
    for folder in folders:
        folder_path = os.path.join(base_dir, folder)
        os.makedirs(folder_path, exist_ok=True)
        print(f"   ✅ Created: {folder}/")
    
    print("\n✅ Folder structure created successfully!")
    return True

def validate_dataset(base_dir):
    """Validate dataset structure and content"""
    print_banner("🔍 VALIDATING DATASET")
    
    issues = []
    warnings = []
    
    # Check if base directory exists
    if not os.path.exists(base_dir):
        issues.append(f"Base directory not found: {base_dir}")
        return issues, warnings
    
    # Check required folders
    required_folders = ['cloudy', 'clear']
    for folder in required_folders:
        folder_path = os.path.join(base_dir, folder)
        if not os.path.exists(folder_path):
            issues.append(f"Missing required folder: {folder}/")
    
    if issues:
        return issues, warnings
    
    # Check image counts
    cloudy_path = os.path.join(base_dir, 'cloudy')
    clear_path = os.path.join(base_dir, 'clear')
    sar_path = os.path.join(base_dir, 'sar')
    
    cloudy_files = [f for f in os.listdir(cloudy_path) if f.lower().endswith(('.jpg', '.png', '.jpeg', '.tif', '.tiff'))]
    clear_files = [f for f in os.listdir(clear_path) if f.lower().endswith(('.jpg', '.png', '.jpeg', '.tif', '.tiff'))]
    
    print(f"📊 Image counts:")
    print(f"   Cloudy images: {len(cloudy_files)}")
    print(f"   Clear images: {len(clear_files)}")
    
    if os.path.exists(sar_path):
        sar_files = [f for f in os.listdir(sar_path) if f.lower().endswith(('.jpg', '.png', '.jpeg', '.tif', '.tiff'))]
        print(f"   SAR images: {len(sar_files)}")
    else:
        print(f"   SAR images: 0 (optional - will use random data)")
        warnings.append("SAR folder not found - will use random SAR data during training")
    
    # Check if counts match
    if len(cloudy_files) != len(clear_files):
        issues.append(f"Mismatch: {len(cloudy_files)} cloudy vs {len(clear_files)} clear images")
    
    # Check if filenames match
    cloudy_set = set(cloudy_files)
    clear_set = set(clear_files)
    
    missing_in_clear = cloudy_set - clear_set
    missing_in_cloudy = clear_set - cloudy_set
    
    if missing_in_clear:
        issues.append(f"{len(missing_in_clear)} files in cloudy/ but not in clear/")
        print(f"\n⚠️  Files in cloudy/ but missing in clear/:")
        for f in list(missing_in_clear)[:5]:
            print(f"     - {f}")
        if len(missing_in_clear) > 5:
            print(f"     ... and {len(missing_in_clear) - 5} more")
    
    if missing_in_cloudy:
        issues.append(f"{len(missing_in_cloudy)} files in clear/ but not in cloudy/")
        print(f"\n⚠️  Files in clear/ but missing in cloudy/:")
        for f in list(missing_in_cloudy)[:5]:
            print(f"     - {f}")
        if len(missing_in_cloudy) > 5:
            print(f"     ... and {len(missing_in_cloudy) - 5} more")
    
    # Check minimum dataset size
    if len(cloudy_files) < 10:
        warnings.append(f"Very small dataset ({len(cloudy_files)} images) - recommend 100+ for good results")
    elif len(cloudy_files) < 100:
        warnings.append(f"Small dataset ({len(cloudy_files)} images) - recommend 500+ for best results")
    
    return issues, warnings

def copy_images_interactive(source_dir, dest_dir, folder_type):
    """Interactive image copying"""
    print(f"\n📋 Copying {folder_type} images...")
    print(f"   Source: {source_dir}")
    print(f"   Destination: {dest_dir}")
    
    if not os.path.exists(source_dir):
        print(f"   ❌ Source directory not found!")
        return False
    
    # Get all image files
    image_extensions = ('.jpg', '.jpeg', '.png', '.tif', '.tiff')
    image_files = [f for f in os.listdir(source_dir) if f.lower().endswith(image_extensions)]
    
    if not image_files:
        print(f"   ⚠️  No images found in source directory")
        return False
    
    print(f"   Found {len(image_files)} images")
    
    # Copy files
    copied = 0
    for img_file in image_files:
        src = os.path.join(source_dir, img_file)
        dst = os.path.join(dest_dir, img_file)
        
        try:
            shutil.copy2(src, dst)
            copied += 1
        except Exception as e:
            print(f"   ⚠️  Error copying {img_file}: {e}")
    
    print(f"   ✅ Copied {copied}/{len(image_files)} images")
    return True

def main():
    parser = argparse.ArgumentParser(description='Prepare dataset for cloud removal training')
    parser.add_argument('--data_dir', type=str, default='./data',
                        help='Base directory for dataset')
    parser.add_argument('--create', action='store_true',
                        help='Create folder structure')
    parser.add_argument('--validate', action='store_true',
                        help='Validate existing dataset')
    parser.add_argument('--interactive', action='store_true',
                        help='Interactive mode for copying images')
    
    args = parser.parse_args()
    
    print_banner("🛰️  DATASET PREPARATION TOOL")
    
    # Create folder structure
    if args.create:
        create_folder_structure(args.data_dir)
    
    # Interactive mode
    if args.interactive:
        print_banner("📂 INTERACTIVE DATASET SETUP")
        
        # Create folders first
        create_folder_structure(args.data_dir)
        
        print("\n📝 You need to provide:")
        print("   1. Cloudy satellite images (with clouds)")
        print("   2. Clear satellite images (same location, no clouds)")
        print("   3. SAR images (optional)")
        
        print("\n⚠️  Important: Filenames must match across folders!")
        print("   Example: cloudy/image_001.jpg ↔ clear/image_001.jpg")
        
        # Ask for source directories
        print("\n" + "-" * 80)
        cloudy_source = input("Enter path to cloudy images folder (or press Enter to skip): ").strip()
        if cloudy_source and os.path.exists(cloudy_source):
            copy_images_interactive(cloudy_source, os.path.join(args.data_dir, 'cloudy'), 'cloudy')
        
        clear_source = input("Enter path to clear images folder (or press Enter to skip): ").strip()
        if clear_source and os.path.exists(clear_source):
            copy_images_interactive(clear_source, os.path.join(args.data_dir, 'clear'), 'clear')
        
        sar_source = input("Enter path to SAR images folder (optional, press Enter to skip): ").strip()
        if sar_source and os.path.exists(sar_source):
            copy_images_interactive(sar_source, os.path.join(args.data_dir, 'sar'), 'SAR')
    
    # Validate dataset
    if args.validate or args.interactive:
        issues, warnings = validate_dataset(args.data_dir)
        
        print("\n" + "=" * 80)
        if not issues and not warnings:
            print("  ✅ DATASET VALIDATION PASSED!")
            print("=" * 80)
            print("\n🎉 Your dataset is ready for training!")
            print("\n📝 Next steps:")
            print("   1. Run: python quick_train.py")
            print("   2. Or: python src/train_cloud_removal.py --epochs 20")
        else:
            if issues:
                print("  ❌ VALIDATION FAILED - ISSUES FOUND")
                print("=" * 80)
                print("\n🔴 Critical Issues:")
                for issue in issues:
                    print(f"   • {issue}")
            
            if warnings:
                if not issues:
                    print("  ⚠️  VALIDATION PASSED WITH WARNINGS")
                    print("=" * 80)
                print("\n⚠️  Warnings:")
                for warning in warnings:
                    print(f"   • {warning}")
            
            if issues:
                print("\n❌ Please fix the issues above before training")
            else:
                print("\n✅ You can proceed with training, but consider the warnings")
    
    # Show usage if no flags
    if not (args.create or args.validate or args.interactive):
        print("📖 Usage Examples:")
        print("\n   Create folder structure:")
        print("   python prepare_dataset.py --create")
        print("\n   Interactive setup:")
        print("   python prepare_dataset.py --interactive")
        print("\n   Validate existing dataset:")
        print("   python prepare_dataset.py --validate")
        print("\n   Custom data directory:")
        print("   python prepare_dataset.py --data_dir /path/to/data --validate")

if __name__ == "__main__":
    main()
