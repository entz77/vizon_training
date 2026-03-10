"""
Data loader utilities for YOLO training
"""

import torch
from torch.utils.data import DataLoader
from .dataset import YOLODataset


def create_dataloader(
    img_dir,
    label_dir,
    batch_size=32,
    img_size=640,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
    augment=False,
    classes=None
):
    """
    Create a DataLoader for YOLO training.
    
    Args:
        img_dir (str): Path to images directory
        label_dir (str): Path to labels directory
        batch_size (int): Batch size
        img_size (int): Target image size
        shuffle (bool): Whether to shuffle the data
        num_workers (int): Number of worker processes
        pin_memory (bool): Whether to pin memory
        augment (bool): Apply data augmentation
        classes (list): List of class names
    
    Returns:
        DataLoader: PyTorch DataLoader instance
    """
    dataset = YOLODataset(
        img_dir=img_dir,
        label_dir=label_dir,
        img_size=img_size,
        classes=classes,
        augment=augment
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn
    )
    
    return dataloader


def collate_fn(batch):
    """
    Custom collate function for batching.
    Handles variable-length target tensors.
    """
    imgs, targets, img_names = zip(*batch)
    
    # Stack images
    imgs = torch.stack(imgs, dim=0)
    
    # Create targets with padding
    batch_targets = []
    for i, target in enumerate(targets):
        if len(target) > 0:
            # Add batch index to each target
            batch_target = torch.cat([
                torch.full((len(target), 1), i),
                target
            ], dim=1)
            batch_targets.append(batch_target)
    
    targets = torch.cat(batch_targets, dim=0) if batch_targets else torch.zeros((0, 6))
    
    return imgs, targets, img_names
