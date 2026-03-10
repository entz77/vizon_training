"""
Configuration validation utility
"""

import yaml
from pathlib import Path


def validate_config(config_path):
    """Validate training configuration"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    required_keys = ['model_size', 'epochs', 'batch_size', 'imgsz']
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required key: {key}")
    
    return config


def validate_dataset_config(config_path):
    """Validate dataset configuration"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    required_keys = ['path', 'train', 'val', 'nc', 'names']
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required key in dataset config: {key}")
    
    # Check if paths exist
    base_path = Path(config['path'])
    if not base_path.exists():
        print(f"Warning: Base dataset path does not exist: {base_path}")
    
    return config


if __name__ == '__main__':
    print("Validating configurations...")
    
    try:
        training_config = validate_config('configs/training_config.yaml')
        print("✓ Training config is valid")
    except Exception as e:
        print(f"✗ Training config error: {e}")
    
    try:
        dataset_config = validate_dataset_config('configs/dataset_config.yaml')
        print("✓ Dataset config is valid")
    except Exception as e:
        print(f"✗ Dataset config error: {e}")
    
    print("\nConfiguration validation complete!")
