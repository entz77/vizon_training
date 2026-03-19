"""
YOLO Model Training Manager
"""

import os
import json
import logging
from pathlib import Path
from datetime import datetime
import torch
import yaml

from ..models import YOLOModel


class YOLOTrainer:
    """
    Complete training pipeline for YOLO models.
    Handles training, logging, checkpointing, and validation.
    """
    
    def __init__(self, config_path=None, log_dir='logs'):
        """
        Initialize trainer.
        
        Args:
            config_path (str): Path to training configuration YAML file
            log_dir (str): Directory for training logs
        """
        self.config = self._load_config(config_path) if config_path else {}
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True, parents=True)
        
        # Setup logging
        self.logger = self._setup_logger()
        
        # Initialize model
        model_size = self.config.get('model_size', 'm')
        self.model = YOLOModel(model_size=model_size)
        
        self.logger.info(f"Trainer initialized with model size: {model_size}")
    
    def _load_config(self, config_path):
        """Load configuration from YAML file"""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _setup_logger(self):
        """Setup logging"""
        logger = logging.getLogger('YOLOTrainer')
        logger.setLevel(logging.INFO)
        
        # File handler
        fh = logging.FileHandler(
            self.log_dir / f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
        )
        fh.setLevel(logging.INFO)
        
        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        logger.addHandler(fh)
        logger.addHandler(ch)
        
        return logger
    
    def train(
        self,
        data_yaml,
        epochs=100,
        batch_size=16,
        imgsz=640,
        patience=20,
        lr0=0.01,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3,
        save_dir='run'
    ):
        """
        Train YOLO model.
        
        Args:
            data_yaml (str): Path to dataset configuration YAML
            epochs (int): Number of training epochs
            batch_size (int): Batch size for training
            imgsz (int): Input image size
            patience (int): Early stopping patience
            lr0 (float): Initial learning rate
            lrf (float): Final learning rate ratio
            momentum (float): SGD momentum
            weight_decay (float): Weight decay
            warmup_epochs (int): Warmup epochs
            save_dir (str): Directory to save training results
        
        Returns:
            dict: Training results
        """
        self.logger.info("=" * 50)
        self.logger.info("Starting YOLO Training")
        self.logger.info("=" * 50)
        self.logger.info(f"Data YAML: {data_yaml}")
        self.logger.info(f"Epochs: {epochs}")
        self.logger.info(f"Batch Size: {batch_size}")
        self.logger.info(f"Image Size: {imgsz}")
        
        # Train
        results = self.model.train(
            data_yaml=data_yaml,
            epochs=epochs,
            batch_size=batch_size,
            imgsz=imgsz,
            patience=patience,
            save_dir=save_dir,
            device=self.model.device,
            lr0=lr0,
            lrf=lrf,
            momentum=momentum,
            weight_decay=weight_decay,
            warmup_epochs=warmup_epochs,
            verbose=True
        )
        
        self.logger.info("Training completed successfully!")
        
        return results
    
    def validate(self, data_yaml, weights_path=None, imgsz=640, batch_size=16):
        """
        Validate model.
        
        Args:
            data_yaml (str): Path to dataset configuration YAML
            weights_path (str): Path to model weights. Uses current if None.
            imgsz (int): Input image size
            batch_size (int): Batch size
        
        Returns:
            dict: Validation results
        """
        if weights_path:
            self.model.load_weights(weights_path)
        
        self.logger.info("Starting validation...")
        results = self.model.val(
            data_yaml=data_yaml,
            imgsz=imgsz,
            batch_size=batch_size
        )
        
        return results
    
    def save_config(self, save_path):
        """Save training configuration"""
        with open(save_path, 'w') as f:
            yaml.dump(self.config, f)
        self.logger.info(f"Config saved to {save_path}")
    
    def log_metrics(self, metrics_dict, step=None):
        """Log training metrics"""
        log_msg = f"Metrics - " + " | ".join(
            [f"{k}: {v:.4f}" for k, v in metrics_dict.items()]
        )
        if step is not None:
            log_msg = f"Step {step} - " + log_msg
        
        self.logger.info(log_msg)
