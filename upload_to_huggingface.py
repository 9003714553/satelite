# -*- coding: utf-8 -*-
 """
Simple Hugging Face Login and Upload Helper
Run this script to login and upload your project
"""

from huggingface_hub import HfApi, login
import os
import sys

# Fix encoding for Windows
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

print("=" * 60)
print("Hugging Face Upload Helper")
print("=" * 60)

# Step 1: Login
print("\nStep 1: Login to Hugging Face")
print("-" * 60)
print("You need a Hugging Face access token.")
print("Get it from: https://huggingface.co/settings/tokens")
print("Make sure to create a 'Write' token!")
print("-" * 60)

token = input("\nPaste your token here (won't be visible): ").strip()

try:
    login(token=token, add_to_git_credential=True)
    print("[SUCCESS] Login successful!")
except Exception as e:
    print(f"[ERROR] Login failed: {e}")
    print("\nPlease check:")
    print("1. Token is correct")
    print("2. Token has 'Write' permission")
    exit(1)

# Step 2: Get repository details
print("\n" + "=" * 60)
print("Step 2: Repository Details")
print("=" * 60)

username = input("Enter your Hugging Face username: ").strip()
repo_name = input("Enter repository name (e.g., satellite-cloud-removal): ").strip()

repo_id = f"{username}/{repo_name}"

print(f"\nRepository: {repo_id}")

# Step 3: Create repository (if needed)
print("\n" + "=" * 60)
print("Step 3: Create Repository")
print("=" * 60)

create_repo = input("Do you want to create the repository? (y/n): ").strip().lower()

api = HfApi()

if create_repo == 'y':
    try:
        api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
        print(f"[SUCCESS] Repository created/verified: https://huggingface.co/{repo_id}")
    except Exception as e:
        print(f"[NOTE] {e}")
        print("Repository might already exist, continuing...")

# Step 4: Upload
print("\n" + "=" * 60)
print("Step 4: Upload Project")
print("=" * 60)

folder_path = r"c:\Users\hp\Downloads\projects AI\Satellite-Imagery-Cloud-Removal\High-Resolution-Satellite-Imagery-Cloud-Removal"

print(f"Folder: {folder_path}")
print(f"Size: ~3.55 GB")
print(f"Destination: {repo_id}")
print("-" * 60)

confirm = input("\nStart upload? This will take time! (y/n): ").strip().lower()

if confirm != 'y':
    print("[CANCELLED] Upload cancelled.")
    exit(0)

print("\n[UPLOADING] Please wait and DO NOT close this window!")
print("=" * 60)

try:
    api.upload_folder(
        folder_path=folder_path,
        repo_id=repo_id,
        repo_type="model",
        commit_message="Upload Satellite Imagery Cloud Removal AI v5.0"
    )
    
    print("\n" + "=" * 60)
    print("[SUCCESS] Upload complete!")
    print("=" * 60)
    print(f"View your project: https://huggingface.co/{repo_id}")
    print("=" * 60)
    
except Exception as e:
    print("\n" + "=" * 60)
    print(f"[ERROR] Upload failed: {e}")
    print("=" * 60)
    print("\nCommon fixes:")
    print("1. Check internet connection")
    print("2. Verify repository exists")
    print("3. Make sure token has Write permission")
    print("4. Try running the script again")
