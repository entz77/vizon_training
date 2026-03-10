"""
YOLO Dataset class for handling object detection data
"""

import os
import json
from pathlib import Path
import cv2
import numpy as np
from torch.utils.data import Dataset
import torch
from PIL import Image


class YOLODataset(Dataset):
    """
    Custom dataset class for YOLO object detection.
    
    Expected directory structure:
    dataset/
        ├── images/
        │   ├── train/
        │   ├── val/
        │   └── test/
        └── labels/
            ├── train/
            ├── val/
            └── test/
    
    Label format (YOLO format):
    <class_id> <x_center> <y_center> <width> <height>
    (normalized coordinates between 0 and 1)
    """
    
    def __init__(self, img_dir, label_dir, img_size=640, classes=None, augment=False):
        """
        Args:
            img_dir (str): Path to images directory
            label_dir (str): Path to labels directory
            img_size (int): Target image size
            classes (list): List of class names
            augment (bool): Apply data augmentation
        """
        self.img_dir = Path(img_dir)
        self.label_dir = Path(label_dir)
        self.img_size = img_size
        self.classes = classes or []
        self.augment = augment
        
        # Get all image files
        self.img_files = sorted([f for f in self.img_dir.glob('*') 
                                if f.suffix.lower() in ['.jpg', '.jpeg', '.png']])
        
        if not self.img_files:
            raise ValueError(f"No images found in {img_dir}")
    
    def __len__(self):
        return len(self.img_files)
    
    def __getitem__(self, idx):
        """
        Returns:
            img (torch.Tensor): Image tensor [3, H, W]
            targets (torch.Tensor): Target tensor [num_objects, 5]
                                   format: [class_id, x_center, y_center, width, height]
        """
        img_path = self.img_files[idx]
        label_path = self.label_dir / img_path.stem / '.txt'
        
        # Load image
        img = cv2.imread(str(img_path))
        if img is None:
            raise ValueError(f"Failed to load image: {img_path}")
        
        h, w = img.shape[:2]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Load labels
        targets = []
        if label_path.exists():
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        targets.append([float(x) for x in parts])
        
        targets = np.array(targets) if targets else np.zeros((0, 5))
        
        # Resize image
        img = cv2.resize(img, (self.img_size, self.img_size))
        
        # Convert to tensor
        img_tensor = torch.from_numpy(img).float().permute(2, 0, 1) / 255.0
        targets_tensor = torch.from_numpy(targets).float()
        
        return img_tensor, targets_tensor, img_path.name
    
    def get_class_names(self):
        """Return class names"""
        return self.classes
    
    @staticmethod
    def load_from_yaml(yaml_path):
        """
        Load dataset configuration from YAML file.
        
        Expected YAML format:
        path: /path/to/dataset
        train: images/train
        val: images/val
        test: images/test
        nc: 80
        names: ['person', 'car', ...]
        """
        import yaml
        
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)
        
        return data
