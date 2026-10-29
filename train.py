"""
Main training script for object detection (backend set by model_type in the training config)
"""

import argparse
import sys
from pathlib import Path
import yaml

# Guarantee the app directory is on sys.path regardless of how the container is launched
APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from src.training import DetectorTrainer
from src.evaluation import build_report
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
    print("Object Detection Training")
    print("=" * 50)
    print(f"Device: {device}")
    print(f"Config: {args.config}")
    print(f"Data YAML: {args.data}")
    print("=" * 50)
    
    # Initialize trainer
    trainer = DetectorTrainer(
        config_path=args.config,
        log_dir=args.log_dir,
        model_type=args.model_type,
        model_size=args.model_size
    )
    
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
    
    # Backend-neutral metrics report (mAP, confusion matrix, PR/F1 curves)
    if args.report:
        best = Path(args.save_dir) / 'checkpoint_best_total.pth'  # rfdetr; yolo keeps best weights loaded
        if trainer.model_type == 'rfdetr' and best.exists():
            trainer.model.load_weights(best)
        report_dir = Path(args.save_dir) / 'report'
        print("\n" + "=" * 50)
        print(f"Building report on '{args.report_split}' split -> {report_dir}")
        print("=" * 50)
        report = build_report(
            trainer.model, args.data, split=args.report_split, out_dir=report_dir,
            conf=args.report_conf, batch_size=args.batch_size,
            num_samples=args.report_samples
        )
        print(f"mAP50-95: {report['map50_95']:.4f} | mAP50: {report['map50']:.4f} | "
              f"best F1: {report['best_f1']:.3f} @ conf {report['best_f1_conf']:.2f}")
        for w in report['warnings']:
            print(f"WARNING: {w}")

    print("\n" + "=" * 50)
    print("Training Completed Successfully!")
    print("=" * 50)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train object detection model'
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
        '--model-type',
        type=str,
        default=None,
        help="Detection backend, overrides 'model_type' in the config (e.g. yolo, rfdetr)"
    )
    parser.add_argument(
        '--model-size',
        type=str,
        default=None,
        help="Model size/variant, overrides 'model_size' in the config"
    )
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
    
    parser.add_argument(
        '--report',
        action='store_true',
        help='After training, write a metrics report (mAP, confusion matrix, PR/F1 curves) to <save-dir>/report'
    )
    parser.add_argument(
        '--report-split',
        type=str,
        default='val',
        choices=['train', 'val', 'test'],
        help='Dataset split for --report'
    )
    parser.add_argument(
        '--report-conf',
        type=float,
        default=0.25,
        help='Confidence threshold for the report confusion matrix and per-class P/R/F1'
    )

    parser.add_argument(
        '--report-samples',
        type=int,
        default=16,
        help='Number of sample images (ground truth vs predictions) to save with --report, 0 to disable'
    )

    args = parser.parse_args()

    # Handle cuda flag
    if args.no_cuda:
        args.use_cuda = False
    
    main(args)
