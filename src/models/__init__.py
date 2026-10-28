from .base import BaseDetector, Detection, ImageDetections
from .registry import build_model, get_model_class, available_models


def __getattr__(name):
    # Lazy so importing src.models doesn't require every backend's dependencies.
    if name == 'YOLOModel':
        return get_model_class('yolo')
    if name == 'RFDETRModel':
        return get_model_class('rfdetr')
    raise AttributeError(name)


__all__ = [
    'BaseDetector', 'Detection', 'ImageDetections',
    'build_model', 'get_model_class', 'available_models',
    'YOLOModel', 'RFDETRModel',
]
