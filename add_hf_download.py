# -*- coding: utf-8 -*-
"""
Add Hugging Face Model Download to app.py
This script adds automatic model downloading from Hugging Face
"""

import os

app_file = r"c:\Users\hp\Downloads\projects AI\Satellite-Imagery-Cloud-Removal\High-Resolution-Satellite-Imagery-Cloud-Removal\src\app.py"

# Code to add at the beginning of app.py (after imports)
model_download_code = '''
# Auto-download models from Hugging Face if not present
def download_models_from_hf():
    """Download model files from Hugging Face if they don't exist locally"""
    from huggingface_hub import hf_hub_download
    import os
    
    repo_id = "VIJAYarajan03/satellite-cloud-removal"
    model_files = {
        'gen_epoch_5.pth': 'Generator model (179.6 MB)',
        'disc_epoch_5.pth': 'Discriminator model (7.6 MB)',
    }
    
    for filename, description in model_files.items():
        if not os.path.exists(filename):
            try:
                print(f"📥 Downloading {description}...")
                hf_hub_download(
                    repo_id=repo_id,
                    filename=filename,
                    local_dir=".",
                    local_dir_use_symlinks=False,
                    cache_dir=".cache"
                )
                print(f"✅ {filename} downloaded!")
            except Exception as e:
                print(f"⚠️ Could not download {filename}: {e}")
                print(f"   Please ensure the file exists at: https://huggingface.co/{repo_id}")
    
    print("🚀 Models ready!")

# Download models on app start
try:
    download_models_from_hf()
except Exception as e:
    print(f"Note: Model download skipped - {e}")
'''

print("=" * 60)
print("Hugging Face Model Download Setup")
print("=" * 60)
print("\nThis will add automatic model downloading to your app.py")
print("So Streamlit Cloud can fetch models from Hugging Face!")
print("\n" + "=" * 60)

# Read current app.py
with open(app_file, 'r', encoding='utf-8') as f:
    content = f.read()

# Check if already added
if 'download_models_from_hf' in content:
    print("\n[INFO] Model download code already exists in app.py")
    print("No changes needed!")
else:
    print("\n[ACTION] Adding model download code...")
    
    # Find a good place to insert (after imports, before main code)
    # Look for "st.set_page_config" or "def main" or first function
    insert_position = content.find('st.set_page_config')
    
    if insert_position == -1:
        insert_position = content.find('def ')
    
    if insert_position == -1:
        # Just add after imports (find first non-import line)
        lines = content.split('\n')
        for i, line in enumerate(lines):
            if line.strip() and not line.strip().startswith('import') and not line.strip().startswith('from') and not line.strip().startswith('#'):
                insert_position = content.find(line)
                break
    
    # Insert the code
    new_content = content[:insert_position] + model_download_code + '\n\n' + content[insert_position:]
    
    # Write back
    with open(app_file, 'w', encoding='utf-8') as f:
        f.write(new_content)
    
    print("[SUCCESS] Model download code added to app.py!")
    print("\nYour app will now automatically download models from Hugging Face")
    print("when deployed to Streamlit Cloud!")

print("\n" + "=" * 60)
print("Next Steps:")
print("=" * 60)
print("1. Check the updated src/app.py")
print("2. Add 'huggingface_hub' to requirements.txt (if not already there)")
print("3. Push to GitHub")
print("4. Deploy to Streamlit Cloud")
print("=" * 60)
