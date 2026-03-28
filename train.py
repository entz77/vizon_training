"""
Main training script for YOLO object detection
"""

import argparse
import sys
from pathlib import Path
import yaml

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.training import YOLOTrainer
from utils import setup_seed, get_device


def load_config_defaults(config_path):
    """Load training config file to use as defaults"""
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Warning: Config file not found at {config_path}")
        return {}


def main(args):
    """Main training function"""
    
    # Setup
    setup_seed(args.seed)
    device = get_device(use_cuda=args.use_cuda)
    
    print("=" * 50)
    print("YOLO Object Detection Training")
    print("=" * 50)
    print(f"Device: {device}")
    print(f"Config: {args.config}")
    print(f"Data YAML: {args.data}")
    print("=" * 50)
    
    # Initialize trainer
    trainer = YOLOTrainer(config_path=args.config, log_dir=args.log_dir)
    
    # Train
    results = trainer.train(
        data_yaml=args.data,
        epochs=args.epochs,
        batch_size=args.batch_size,
        imgsz=args.imgsz,
        patience=args.patience,
        lr0=args.lr0,
        lrf=args.lrf,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs,
        save_dir=args.save_dir
    )
    
    # Validation
    if args.validate:
        print("\n" + "=" * 50)
        print("Starting Validation")
        print("=" * 50)
        val_results = trainer.validate(
            data_yaml=args.data,
            imgsz=args.imgsz,
            batch_size=args.batch_size
        )
        print("Validation completed!")
    
    print("\n" + "=" * 50)
    print("Training Completed Successfully!")
    print("=" * 50)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train YOLO object detection model'
    )
    
    # Data arguments
    parser.add_argument(
        '--data',
        type=str,
        default='configs/dataset_config.yaml',
        help='Path to dataset configuration YAML file'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='configs/training_config.yaml',
        help='Path to training configuration YAML file'
    )
    
    # Parse config argument first to load defaults
    args, remaining = parser.parse_known_args()
    config = load_config_defaults(args.config)
    
    # Now add remaining arguments with defaults from config
    parser.add_argument(
        '--epochs',
        type=int,
        default=config.get('epochs', 100),
        help='Number of training epochs'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=config.get('batch_size', 32),
        help='Batch size for training'
    )
    parser.add_argument(
        '--imgsz',
        type=int,
        default=config.get('imgsz', 640),
        help='Input image size'
    )
    parser.add_argument(
        '--patience',
        type=int,
        default=config.get('patience', 20),
        help='Early stopping patience'
    )
    
    # Learning rate arguments
    parser.add_argument(
        '--lr0',
        type=float,
        default=config.get('lr0', 0.01),
        help='Initial learning rate'
    )
    parser.add_argument(
        '--lrf',
        type=float,
        default=config.get('lrf', 0.01),
        help='Final learning rate ratio'
    )
    
    # Optimizer arguments
    parser.add_argument(
        '--momentum',
        type=float,
        default=config.get('momentum', 0.937),
        help='SGD momentum'
    )
    parser.add_argument(
        '--weight-decay',
        type=float,
        default=config.get('weight_decay', 0.0005),
        help='Weight decay'
    )
    parser.add_argument(
        '--warmup-epochs',
        type=int,
        default=config.get('warmup_epochs', 3),
        help='Warmup epochs'
    )
    
    # Output arguments
    parser.add_argument(
        '--save-dir',
        type=str,
        default=config.get('output_dir', 'runs/train'),
        help='Directory to save training results'
    )
    parser.add_argument(
        '--log-dir',
        type=str,
        default=config.get('log_dir', 'logs'),
        help='Directory to save logs'
    )
    
    # General arguments
    parser.add_argument(
        '--use-cuda',
        action='store_true',
        default=True,
        help='Use CUDA if available'
    )
    parser.add_argument(
        '--no-cuda',
        action='store_true',
        help='Do not use CUDA'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )
    parser.add_argument(
        '--validate',
        action='store_true',
        help='Run validation after training'
    )
    
    args = parser.parse_args()
    
    # Handle cuda flag
    if args.no_cuda:
        args.use_cuda = False
    
    main(args)
