"""
YOLO Model wrapper using Ultralytics YOLO26
"""

from ultralytics import YOLO
from typing import List, Optional, Dict

from .base import BaseDetector, Detection, ImageDetections


class YOLOModel(BaseDetector):
    """
    Wrapper class for Ultralytics YOLO model.
    Provides easy interface for loading, training, and inference.
    """

    model_type = 'yolo'
    default_size = 'm'

    def __init__(self, model_size='m', device=None, class_id_map: Optional[Dict[int, int]] = None):
        """
        Initialize YOLO model.
        
        Args:
            model_size (str): Model size - 'n' (nano), 's' (small), 'm' (medium),
                            'l' (large), 'x' (extra large)
            device (str): Device to use - 'cuda' or 'cpu'. Auto-detected if None.
            class_id_map (dict): Optional mapping of class IDs. Maps original class IDs to new class IDs.
        """
        super().__init__(model_size=model_size, device=device, class_id_map=class_id_map)

        # Initialize model
        self.model = YOLO(f'yolo26{self.model_size}.pt')
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
    
    def predict(self, source, conf=0.25, iou=0.45, imgsz=640, device=None, verbose=True) -> List[ImageDetections]:
        """
        Run inference on images or video.

        Args:
            source (str): Path to image/video or PIL Image
            conf (float): Confidence threshold
            iou (float): IOU threshold for NMS
            imgsz (int): Input image size
            device (str): Device to use
            verbose (bool): Print per-image inference logs

        Returns:
            list[ImageDetections]: One entry per image
        """
        device = device or self.device

        results = self.model.predict(
            source=source,
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            device=device,
            verbose=verbose
        )

        return [self._to_image_detections(r) for r in results]

    @staticmethod
    def _to_image_detections(result) -> ImageDetections:
        detections = []
        if result.boxes is not None:
            for box in result.boxes:
                class_id = int(box.cls[0])
                x1, y1, x2, y2 = (float(v) for v in box.xyxy[0])
                detections.append(Detection(
                    class_id=class_id,
                    class_name=result.names[class_id],
                    confidence=float(box.conf[0]),
                    x1=x1, y1=y1, x2=x2, y2=y2,
                ))
        return ImageDetections(
            image_path=getattr(result, 'path', None),
            image_shape=tuple(result.orig_shape) if getattr(result, 'orig_shape', None) is not None else None,
            detections=detections,
        )
    
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
        Automatically loads accompanying mapping file if it exists (e.g., best.pt → best.mapping.yaml)
        
        Args:
            weights_path (str): Path to weights file
        """
        self.model = YOLO(weights_path)
        self.model.to(self.device)
        self._load_sibling_mapping(weights_path)
