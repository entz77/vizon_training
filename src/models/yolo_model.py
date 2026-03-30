"""
YOLO Model wrapper using Ultralytics YOLO26
"""

import torch
from ultralytics import YOLO
from pathlib import Path


class YOLOModel:
    """
    Wrapper class for Ultralytics YOLO model.
    Provides easy interface for loading, training, and inference.
    """
    
    def __init__(self, model_size='m', device=None):
        """
        Initialize YOLO model.
        
        Args:
            model_size (str): Model size - 'n' (nano), 's' (small), 'm' (medium),
                            'l' (large), 'x' (extra large)
            device (str): Device to use - 'cuda' or 'cpu'. Auto-detected if None.
        """
        self.model_size = model_size
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize model
        self.model = YOLO(f'yolo26{model_size}.pt')
        self.model.to(self.device)
    
    def train(
        self,
        data_yaml,
        epochs=100,
        batch_size=16,
        imgsz=640,
        patience=20,
        save_dir='train',
        resume=False,
        device=None,
        **kwargs
    ):
        """
        Train the YOLO model.
        
        Args:
            data_yaml (str): Path to dataset YAML file
            epochs (int): Number of training epochs
            batch_size (int): Batch size for training
            imgsz (int): Input image size
            patience (int): Early stopping patience
            save_dir (str): Directory to save results (uses project/name for auto-incrementing)
            resume (bool): Resume from last checkpoint
            device (str): Device to use for training
            **kwargs: Additional training arguments
        
        Returns:
            dict: Training results
        """
        device = device or self.device
        
        # Use project and name for auto-incrementing instead of save_dir
        # This will create runs/train, runs/train1, runs/train2, etc.
        results = self.model.train(
            data=data_yaml,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch_size,
            patience=patience,
            device=device,
            project='runs',
            name=save_dir,
            resume=resume,
            **kwargs
        )
        
        return results
    
    def val(self, data_yaml, imgsz=640, batch_size=16, device=None):
        """
        Validate the model.
        
        Args:
            data_yaml (str): Path to dataset YAML file
            imgsz (int): Input image size
            batch_size (int): Batch size
            device (str): Device to use
        
        Returns:
            dict: Validation results
        """
        device = device or self.device
        
        results = self.model.val(
            data=data_yaml,
            imgsz=imgsz,
            batch=batch_size,
            device=device
        )
        
        return results
    
    def predict(self, source, conf=0.25, iou=0.45, imgsz=640, device=None):
        """
        Run inference on images or video.
        
        Args:
            source (str): Path to image/video or PIL Image
            conf (float): Confidence threshold
            iou (float): IOU threshold for NMS
            imgsz (int): Input image size
            device (str): Device to use
        
        Returns:
            list: Prediction results
        """
        device = device or self.device
        
        results = self.model.predict(
            source=source,
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            device=device
        )
        
        return results
    
    def export(self, format='onnx', half=False):
        """
        Export model to different formats.
        
        Args:
            format (str): Export format - 'onnx', 'torchscript', 'tflite', etc.
            half (bool): Use half precision
        
        Returns:
            str: Path to exported model
        """
        return self.model.export(format=format, half=half)
    
    def load_weights(self, weights_path):
        """
        Load pretrained weights.
        
        Args:
            weights_path (str): Path to weights file
        """
        self.model = YOLO(weights_path)
        self.model.to(self.device)
    
    def get_model_info(self):
        """Get model information"""
        return {
            'model_size': self.model_size,
            'device': self.device,
            'model': str(self.model)
        }
