"""
Utility functions for YOLO training
"""

import os
import random
import numpy as np
import torch
import yaml
from pathlib import Path


def setup_seed(seed=42):
    """
    Set random seed for reproducibility.
    
    Args:
        seed (int): Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    # For deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(use_cuda=True):
    """
    Get device (CPU or CUDA).
    
    Args:
        use_cuda (bool): Prefer CUDA if available
    
    Returns:
        torch.device: PyTorch device object
    """
    if use_cuda and torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("Using CPU device")
    
    return device


def load_yaml(path):
    """
    Load YAML configuration file.
    
    Args:
        path (str): Path to YAML file
    
    Returns:
        dict: Loaded configuration
    """
    with open(path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def save_yaml(data, path):
    """
    Save data to YAML file.
    
    Args:
        data (dict): Data to save
        path (str): Path to save YAML file
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False)


def create_directories(paths):
    """
    Create multiple directories.
    
    Args:
        paths (list): List of directory paths
    """
    for path in paths:
        Path(path).mkdir(parents=True, exist_ok=True)


def get_latest_checkpoint(checkpoint_dir):
    """
    Get the latest checkpoint from a directory.
    
    Args:
        checkpoint_dir (str): Directory containing checkpoints
    
    Returns:
        str: Path to latest checkpoint or None
    """
    checkpoint_dir = Path(checkpoint_dir)
    
    if not checkpoint_dir.exists():
        return None
    
    checkpoints = sorted(checkpoint_dir.glob('*.pt'))
    
    return str(checkpoints[-1]) if checkpoints else None


def print_gpu_info():
    """Print GPU information"""
    if torch.cuda.is_available():
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"GPU Count: {torch.cuda.device_count()}")
        print(f"Current GPU: {torch.cuda.current_device()}")
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print("No CUDA GPU available")
