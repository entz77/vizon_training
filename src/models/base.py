"""
Backend-neutral detector interface shared by every model implementation
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch


@dataclass
class Detection:
    """A single detected box in absolute pixel coordinates."""
    class_id: int
    class_name: str
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def xyxy(self) -> Tuple[float, float, float, float]:
        return (self.x1, self.y1, self.x2, self.y2)


@dataclass
class ImageDetections:
    """All detections for one image. `image_shape` is (height, width)."""
    image_path: Optional[str]
    image_shape: Optional[Tuple[int, int]]
    detections: List[Detection] = field(default_factory=list)


class BaseDetector(ABC):
    """
    Common interface for detection backends (YOLO, RF-DETR, ...).

    Subclasses set `model_type` / `default_size`, build `self.model` in __init__,
    and implement train / val / predict / export / load_weights.
    """

    model_type: str = ''
    default_size: str = ''

    def __init__(self, model_size=None, device=None, class_id_map: Optional[Dict[int, int]] = None):
        self.model_size = model_size or self.default_size
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.class_id_map = class_id_map
        self.model = None

    @abstractmethod
    def train(self, data_yaml, epochs=100, batch_size=16, imgsz=640, patience=20,
              save_dir='run', resume=False, device=None, **kwargs):
        """Train on the dataset described by `data_yaml`."""

    @abstractmethod
    def val(self, data_yaml, imgsz=640, batch_size=16, device=None):
        """Validate on the dataset described by `data_yaml`. May raise NotImplementedError."""

    @abstractmethod
    def predict(self, source, conf=0.25, iou=0.45, imgsz=640, device=None,
                verbose=True) -> List[ImageDetections]:
        """Run inference. Returns one ImageDetections per image."""

    @abstractmethod
    def export(self, format='onnx', half=False):
        """Export the model. Returns the exported path."""

    @abstractmethod
    def load_weights(self, weights_path):
        """Load trained weights (call `_load_sibling_mapping` afterwards)."""

    def _load_sibling_mapping(self, weights_path):
        """Auto-load a mapping file next to the weights (best.pt -> best.mapping.yaml)."""
        weights_path = Path(weights_path)
        mapping_file = weights_path.parent / f"{weights_path.stem}.mapping.yaml"
        if mapping_file.exists():
            print(f"Auto-loading mapping file: {mapping_file}")
            self.load_class_id_map(str(mapping_file))

    def load_class_id_map(self, yaml_path):
        """
        Load class ID mapping from a YAML file and use it for predictions.

        Example YAML format:
            class_id_map:
              0: 2
              1: 1
              2: 0
        """
        from src.data.label_converter import load_class_id_map_from_yaml
        self.class_id_map = load_class_id_map_from_yaml(Path(yaml_path))
        return self.class_id_map

    def get_model_info(self):
        """Get model information"""
        return {
            'model_type': self.model_type,
            'model_size': self.model_size,
            'device': self.device,
            'model': str(self.model),
            'class_id_map': self.class_id_map,
        }
