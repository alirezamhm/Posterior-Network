import torch
from torchvision import datasets
import numpy as np

class CustomDataset(datasets.VisionDataset):
    def __init__(self, images_path, labels_path, transform=None):
        super().__init__(images_path, transform=transform)
        
        self.images = np.load(f"{images_path}.npy")
        self.labels = np.load(f"{labels_path}.npy")
        self.transform = transform

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, index):
        image, label = self.images[index], self.labels[index]
        if self.transform:
            image = self.transform(image)
        label = torch.tensor(label, dtype=torch.long)
        return image, label