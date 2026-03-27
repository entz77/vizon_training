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
    
    def __init__(self, model_size='m', device=None, task='detect', model_name=None):
        """
        Initialize YOLO model.
        
        Args:
            model_size (str): Model size - 'n' (nano), 's' (small), 'm' (medium),
                            'l' (large), 'x' (extra large)
            device (str): Device to use - 'cuda' or 'cpu'. Auto-detected if None.
            task (str): YOLO task type - 'detect' or 'obb'.
            model_name (str): Optional explicit checkpoint filename/path.
        """
        if task not in {'detect', 'obb'}:
            raise ValueError("task must be either 'detect' or 'obb'")

        self.model_size = model_size
        self.task = task
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_name = model_name or self._build_default_model_name(model_size, task)
        
        # Initialize model
        self.model = YOLO(self.model_name, task=task)
        self.model.to(self.device)

    @staticmethod
    def _build_default_model_name(model_size, task):
        """Build default checkpoint name for the selected task."""
        suffix = '-obb' if task == 'obb' else ''
        return f'yolo26{model_size}{suffix}.pt'
    
    def train(
        self,
        data_yaml,
        epochs=100,
        batch_size=16,
        imgsz=640,
        patience=20,
        save_dir='runs/detect/train',
        resume=False,
        device=None,
        task=None,
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
            save_dir (str): Directory to save results
            resume (bool): Resume from last checkpoint
            device (str): Device to use for training
            task (str): Optional task override ('detect' or 'obb')
            **kwargs: Additional training arguments
        
        Returns:
            dict: Training results
        """
        device = device or self.device
        task = task or self.task
        
        results = self.model.train(
            data=data_yaml,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch_size,
            patience=patience,
            device=device,
            project=str(Path(save_dir).parent),
            name=Path(save_dir).name,
            resume=resume,
            task=task,
            **kwargs
        )
        
        return results
    
    def val(self, data_yaml, imgsz=640, batch_size=16, device=None, task=None):
        """
        Validate the model.
        
        Args:
            data_yaml (str): Path to dataset YAML file
            imgsz (int): Input image size
            batch_size (int): Batch size
            device (str): Device to use
            task (str): Optional task override ('detect' or 'obb')
        
        Returns:
            dict: Validation results
        """
        device = device or self.device
        task = task or self.task
        
        results = self.model.val(
            data=data_yaml,
            imgsz=imgsz,
            batch=batch_size,
            device=device,
            task=task
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
            device=device,
            task=self.task
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
        self.model_name = str(weights_path)
        self.model = YOLO(weights_path, task=self.task)
        self.model.to(self.device)
    
    def get_model_info(self):
        """Get model information"""
        return {
            'model_size': self.model_size,
            'task': self.task,
            'model_name': self.model_name,
            'device': self.device,
            'model': str(self.model)
        }
