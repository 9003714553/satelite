import torch
import os
from torch.utils.data import Dataset, DataLoader
import numpy as np

class CloudRemovalDataset(Dataset):
    def __init__(self, root_dir=None, split='train', transform=None, mock_data=False):
        """
        Args:
            root_dir (string): Directory with all the images.
            split (string): 'train' or 'val'.
            transform (callable, optional): Optional transform to be applied on a sample.
            mock_data (bool): If True, generate random noise instead of loading files. 
                              Useful for testing the pipeline without downloading dataset.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.mock_data = mock_data
        self.files = []  # Initialize to empty list by default
        if not mock_data:
            # Try to find data in common locations
            self.data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
            if not os.path.exists(self.data_dir) and root_dir:
                self.data_dir = root_dir
            
            # If data directory exists, list files. (Simplified assumption: filenames match in subfolders)
            # Structure expected: data/cloudy, data/sar, data/clear
            if self.data_dir and os.path.exists(os.path.join(self.data_dir, 'cloudy')):
                 self.files = sorted(os.listdir(os.path.join(self.data_dir, 'cloudy')))
            else:
                 self.files = []
                 print("No data found. Swapping to mock data would be safer, but for now files list is empty.")

        # In a real scenario, you would list files here
        if len(self.files) == 0 and not mock_data:
             print("Warning: No real data found! Using mock data to prevent crash.")
             self.mock_data = True
             self.files = range(100)
        elif mock_data:
             self.files = range(100)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        if self.mock_data:
            # Generate mock data
            # RGB Cloudy Image (3 channels) normalized -1 to 1
            cloudy_img = torch.randn(3, 256, 256)
            # SAR Image (1 channel)
            sar_img = torch.randn(1, 256, 256)
            # Ground Truth Clear Image (3 channels)
            clear_img = torch.randn(3, 256, 256)
            
            return {
                'cloudy': cloudy_img,
                'sar': sar_img,
                'clear': clear_img
            }
        
        # Real data loading logic
        img_name = self.files[idx]
        from PIL import Image
        import torchvision.transforms as T
        
        cloudy_path = os.path.join(self.data_dir, 'cloudy', img_name)
        sar_path = os.path.join(self.data_dir, 'sar', img_name)
        clear_path = os.path.join(self.data_dir, 'clear', img_name)
        
        # Helper transform
        tr = T.Compose([T.Resize((256, 256)), T.ToTensor(), T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        tr_sar = T.Compose([T.Resize((256, 256)), T.ToTensor(), T.Normalize((0.5,), (0.5,))])
        
        c_img = Image.open(cloudy_path).convert('RGB')
        s_img = Image.open(sar_path).convert('L') # SAR usually grayscale
        gt_img = Image.open(clear_path).convert('RGB')
        
        return {
            'cloudy': tr(c_img),
            'sar': tr_sar(s_img),
            'clear': tr(gt_img)
        }

def get_dataloader(batch_size=4, split='train', mock_data=True):
    dataset = CloudRemovalDataset(split=split, mock_data=mock_data)
    return DataLoader(dataset, batch_size=batch_size, shuffle=(split=='train'))

if __name__ == "__main__":
    loader = get_dataloader()
    batch = next(iter(loader))
    print("Cloudy shape:", batch['cloudy'].shape)
    print("SAR shape:", batch['sar'].shape)
    print("Clear shape:", batch['clear'].shape)
