# How to Run in Google Colab

This guide explains how to run the **High-Resolution Satellite Imagery Cloud Removal** project using Google Colab.

## Prerequisites
1. A Google Account.
2. A free [ngrok](https://ngrok.com/) account (for the tunnel to view the app).

## Steps

### 1. Upload Project to Google Drive
1. Download this entire project folder to your computer.
2. Go to [Google Drive](https://drive.google.com/).
3. Upload the folder **`High-Resolution-Satellite-Imagery-Cloud-Removal`** to your Drive.
   - It is recommended to place it in `MyDrive` (the root folder) so the path is simple.

### 2. Get the Notebook
1. In the project folder you just downloaded/uploaded, locate the file named **`Cloud_Removal_Colab.ipynb`**.
2. Right-click on it in Google Drive > Open with > **Google Colaboratory**.
   - *If you don't see Google Colaboratory, install it from "Connect more apps".*

### 3. Run the Notebook
Follow the steps inside the notebook cells:

1.  **Mount Drive**: Run the first cell to give Colab access to your files.
2.  **Navigate**:
    - **Crucial Step**: Check the `project_path` in the second cell.
    - If you uploaded the folder directly to "My Drive", the default path `/content/drive/MyDrive/High-Resolution-Satellite-Imagery-Cloud-Removal` should work.
    - If you put it inside another folder (e.g., `MyDrive/Projects/`), update the path accordingly.
3.  **Install Dependencies**: Run the cell to install `streamlit`, `pyngrok`, etc.
4.  **Authenticate ngrok**:
    - Get your Authtoken from your [ngrok dashboard](https://dashboard.ngrok.com/get-started/your-authtoken).
    - Paste it into the code where it says `YOUR_AUTHTOKEN_HERE`.
5.  **Run App**: Execute the final cell. It will generate a public URL (e.g., `https://random-id.ngrok-free.app`). Click it to open your Streamlit app!

## Troubleshooting
- **"ModuleNotFoundError"**: This usually means the `os.chdir(project_path)` step failed or pointed to the wrong folder. Verify your folder structure in the Files tab (left sidebar in Colab).
- **"ngrok error"**: Ensure you copied the auth token correctly and that you don't have multiple ngrok sessions running (killing the cell and re-running usually helps).
