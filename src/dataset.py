import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torch

class ChangeDetectionDataset(Dataset):
    """
    Dataset class for Change Detection.
    Assumes a directory structure:
    root/
        A/ (images for time t1)
        B/ (images for time t2)
        label/ (ground truth change masks, optional)
    
    Filenames in A, B, and label should match.
    """
    def __init__(self, root_dir, transform=None, mode='train'):
        self.root_dir = root_dir
        self.mode = mode
        self.transform = transform
        
        self.dir_A = os.path.join(root_dir, 'A')
        self.dir_B = os.path.join(root_dir, 'B')
        self.dir_label = os.path.join(root_dir, 'label')
        
        # If directories don't exist, we might be in a 'demo' mode or just initializing
        if os.path.exists(self.dir_A):
            self.image_names = sorted(os.listdir(self.dir_A))
        else:
            self.image_names = []

        if self.transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                # Normalize for standard ResNet models if needed, else just 0-1
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

    def __len__(self):
        # Return a dummy length if no data is found, to allow testing code to run
        if len(self.image_names) == 0:
            return 10 
        return len(self.image_names)

    def __getitem__(self, idx):
        if len(self.image_names) == 0:
            # Generate dummy data for testing purposes
            img_A = torch.randn(3, 256, 256)
            img_B = torch.randn(3, 256, 256)
            # Random binary mask
            mask = torch.randint(0, 2, (1, 256, 256)).float()
            return img_A, img_B, mask

        img_name = self.image_names[idx]
        path_A = os.path.join(self.dir_A, img_name)
        path_B = os.path.join(self.dir_B, img_name)
        
        image_A = Image.open(path_A).convert('RGB')
        image_B = Image.open(path_B).convert('RGB')
        
        if self.transform:
            image_A = self.transform(image_A)
            image_B = self.transform(image_B)
            
        mask = None
        if self.mode == 'train' or self.mode == 'val':
            path_label = os.path.join(self.dir_label, img_name)
            if os.path.exists(path_label):
                mask = Image.open(path_label).convert('L') # Grayscale
                # Resize mask safely
                temp_transform = transforms.Compose([
                    transforms.Resize((256, 256), interpolation=transforms.InterpolationMode.NEAREST),
                    transforms.ToTensor()
                ])
                mask = temp_transform(mask)
            else:
                 # If label missing in train mode, return zeros or error. 
                 # Here we return zeros for robustness
                 mask = torch.zeros((1, 256, 256))

        if mask is None:
             # For inference mode, we might not have masks
             return image_A, image_B
        
        return image_A, image_B, mask
