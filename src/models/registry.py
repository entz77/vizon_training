"""
Registry of available detection backends.

To add one: subclass BaseDetector in src/models/<name>_model.py and add it below.
Backends are imported lazily so a missing optional dependency only affects that backend.
"""

from importlib import import_module

# model_type -> (module, class name)
_BACKENDS = {
    'yolo': ('src.models.yolo_model', 'YOLOModel'),
    'rfdetr': ('src.models.rfdetr_model', 'RFDETRModel'),
}


def available_models():
    return list(_BACKENDS)


def get_model_class(model_type):
    key = str(model_type).lower()
    if key not in _BACKENDS:
        raise ValueError(f"Unknown model_type '{model_type}'. Available: {', '.join(_BACKENDS)}")
    module, cls = _BACKENDS[key]
    return getattr(import_module(module), cls)


def build_model(model_type, model_size=None, device=None, **kwargs):
    """Instantiate a backend. `model_size=None` uses the backend's default."""
    cls = get_model_class(model_type)
    return cls(model_size=model_size, device=device, **kwargs)
