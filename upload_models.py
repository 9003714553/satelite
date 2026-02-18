# -*- coding: utf-8 -*-
"""
Upload Model Files - Force Upload (Bypass .gitignore)
"""

from huggingface_hub import HfApi, upload_file
import os

repo_id = "VIJAYarajan03/satellite-cloud-removal"
project_folder = r"c:\Users\hp\Downloads\projects AI\Satellite-Imagery-Cloud-Removal\High-Resolution-Satellite-Imagery-Cloud-Removal"

print("=" * 60)
print("Force Upload Model Files (Bypass .gitignore)")
print("=" * 60)

api = HfApi()

# List of model files
models = {
    "gen_epoch_5.pth": "Generator model (epoch 5)",
    "disc_epoch_5.pth": "Discriminator model (epoch 5)",
    "disc_epoch_10.pth": "Discriminator model (epoch 10)",
}

for filename, description in models.items():
    filepath = os.path.join(project_folder, filename)
    
    if not os.path.exists(filepath):
        print(f"\n[SKIP] {filename} - File not found")
        continue
    
    size_mb = os.path.getsize(filepath) / (1024 * 1024)
    print(f"\n[UPLOADING] {filename}")
    print(f"  Description: {description}")
    print(f"  Size: {size_mb:.1f} MB")
    print(f"  This may take several minutes...")
    
    try:
        # Use upload_file with explicit parameters
        url = upload_file(
            path_or_fileobj=filepath,
            path_in_repo=filename,
            repo_id=repo_id,
            repo_type="model",
            commit_message=f"Add {description}",
            token=None,  # Use cached token
        )
        print(f"[SUCCESS] Uploaded to: {url}")
    except Exception as e:
        print(f"[ERROR] Failed: {str(e)}")

print("\n" + "=" * 60)
print("Upload Complete!")
print("=" * 60)
print(f"Check your repository:")
print(f"https://huggingface.co/{repo_id}")
print("=" * 60)
