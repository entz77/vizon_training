"""
RF-DETR model wrapper (https://github.com/roboflow/rf-detr)
"""

import os
import shutil
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import yaml
from PIL import Image

from .base import BaseDetector, Detection, ImageDetections

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
SIZES = ('nano', 'small', 'medium', 'base', 'large')


def _model_class(size):
    import rfdetr
    names = {
        'nano': 'RFDETRNano', 'small': 'RFDETRSmall', 'medium': 'RFDETRMedium',
        'base': 'RFDETRBase', 'large': 'RFDETRLarge',
    }
    return getattr(rfdetr, names[size])


class RFDETRModel(BaseDetector):
    """
    Wrapper class for RF-DETR with the same interface as YOLOModel.
    """

    model_type = 'rfdetr'
    default_size = 'base'

    def __init__(self, model_size='base', device=None, class_id_map: Optional[Dict[int, int]] = None):
        """
        Args:
            model_size (str): 'nano', 'small', 'medium', 'base' or 'large'
            device (str): 'cuda' or 'cpu'. Auto-detected if None.
            class_id_map (dict): Optional mapping of class IDs.
        """
        super().__init__(model_size=model_size, device=device, class_id_map=class_id_map)
        if self.model_size not in SIZES:
            raise ValueError(f"Invalid rfdetr model_size '{self.model_size}'. Choose from {SIZES}")
        self.class_names: Dict[int, str] = {}
        self.model = _model_class(self.model_size)(device=self.device)

    @staticmethod
    def _link_tree(src_dir, dst_dir):
        dst_dir.mkdir(parents=True)
        for f in src_dir.rglob('*'):
            if not f.is_file():
                continue
            dst = dst_dir / f.relative_to(src_dir)
            dst.parent.mkdir(parents=True, exist_ok=True)
            try:
                os.link(f, dst)
            except OSError:
                shutil.copy2(f, dst)

    @staticmethod
    def _prepare_dataset(data_yaml, out_dir):
        """
        RF-DETR expects <root>/{train,valid,test}/{images,labels} + data.yaml, while the
        project's datasets use Ultralytics' images/<split> + labels/<split> layout.
        Build the expected layout out of real directories holding hard links (copies if
        linking fails): rfdetr rejects symlinks that resolve outside the dataset dir.
        """
        with open(data_yaml) as f:
            cfg = yaml.safe_load(f)
        root = Path(cfg.get('path', Path(data_yaml).parent)).resolve()

        out_dir = Path(out_dir)
        if out_dir.exists():
            shutil.rmtree(out_dir)
        for src_key, dst in (('train', 'train'), ('val', 'valid'), ('test', 'test')):
            if not cfg.get(src_key):
                continue
            img_dir = (root / cfg[src_key]).resolve()
            lbl_dir = Path(str(img_dir).replace('/images/', '/labels/'))
            if not img_dir.is_dir():
                if src_key == 'test':
                    continue
                raise FileNotFoundError(f"{src_key} images not found: {img_dir}")
            RFDETRModel._link_tree(img_dir, out_dir / dst / 'images')
            if lbl_dir.is_dir():
                RFDETRModel._link_tree(lbl_dir, out_dir / dst / 'labels')

        names = cfg['names']
        with open(out_dir / 'data.yaml', 'w') as f:
            yaml.dump({
                'path': str(out_dir),
                'train': 'train/images',
                'val': 'valid/images',
                **({'test': 'test/images'} if (out_dir / 'test').exists() else {}),
                'nc': len(names),
                'names': names,
            }, f)
        return out_dir, names

    def train(
        self,
        data_yaml,
        epochs=100,
        batch_size=16,
        imgsz=640,
        patience=20,
        save_dir='run',
        resume=False,
        device=None,
        **kwargs
    ):
        """
        Train the RF-DETR model. `imgsz` is ignored (resolution is fixed per model size);
        extra kwargs (e.g. lr, grad_accum_steps) go straight to rfdetr's train().
        Weights are written to `save_dir` (checkpoint_best_total.pth, ...).
        """
        save_dir = Path(save_dir)
        # Kept outside save_dir so the linked/copied images never end up in the run outputs
        work_dir = Path(tempfile.mkdtemp(prefix='rfdetr_dataset_'))
        try:
            dataset_dir, names = self._prepare_dataset(data_yaml, work_dir / 'dataset')
            self.class_names = {int(k): v for k, v in dict(names).items()} if isinstance(names, dict) \
                else dict(enumerate(names))

            train_args = dict(
                dataset_dir=str(dataset_dir),
                epochs=epochs,
                batch_size=batch_size,
                output_dir=str(save_dir),
                early_stopping=patience is not None and patience > 0,
                early_stopping_patience=patience,
                device=device or self.device,
            )
            if resume:
                last = save_dir / 'checkpoint.pth'
                if last.exists():
                    train_args['resume'] = str(last)
            train_args.update(kwargs)
            return self.model.train(**train_args)
        finally:
            shutil.rmtree(work_dir, ignore_errors=True)

    def val(self, data_yaml, imgsz=640, batch_size=16, device=None):
        raise NotImplementedError(
            "RF-DETR validates every epoch during train(); use evaluate.py for a separate run."
        )

    def predict(self, source, conf=0.25, iou=0.45, imgsz=640, device=None, verbose=True) -> List[ImageDetections]:
        """
        Run inference on image path(s), a directory of images, or numpy frame(s) (BGR, as from cv2).
        `iou`/`imgsz` are ignored: RF-DETR is NMS-free with a fixed resolution.
        """
        images, paths = self._load_images(source)
        if not images:
            return []

        out = []
        for image, path in zip(images, paths):
            dets = self.model.predict(image, threshold=conf)
            out.append(self._to_image_detections(dets, path, image.size))
        return out

    @staticmethod
    def _load_images(source):
        if isinstance(source, (str, Path)):
            p = Path(source)
            source = sorted(str(f) for f in p.rglob('*') if f.suffix.lower() in IMAGE_EXTENSIONS) \
                if p.is_dir() else [str(p)]
        elif isinstance(source, (np.ndarray, Image.Image)):
            source = [source]

        images, paths = [], []
        for item in source:
            if isinstance(item, np.ndarray):
                images.append(Image.fromarray(item[..., ::-1]))  # BGR -> RGB
                paths.append(None)
            elif isinstance(item, Image.Image):
                images.append(item.convert('RGB'))
                paths.append(None)
            else:
                images.append(Image.open(item).convert('RGB'))
                paths.append(str(item))
        return images, paths

    def _name(self, class_id):
        names = self.class_names
        if not names:
            names = getattr(self.model, 'class_names', None) or {}
        if isinstance(names, dict):
            return str(names.get(class_id, class_id))
        return str(names[class_id]) if 0 <= class_id < len(names) else str(class_id)

    def _to_image_detections(self, dets, path, size) -> ImageDetections:
        width, height = size
        detections = []
        for (x1, y1, x2, y2), cid, score in zip(dets.xyxy, dets.class_id, dets.confidence):
            cid = int(cid)
            detections.append(Detection(
                class_id=cid, class_name=self._name(cid), confidence=float(score),
                x1=float(x1), y1=float(y1), x2=float(x2), y2=float(y2),
            ))
        return ImageDetections(image_path=path, image_shape=(height, width), detections=detections)

    def export(self, format='onnx', half=False):
        if format != 'onnx':
            raise ValueError("RF-DETR export supports only 'onnx'")
        return self.model.export()

    def load_weights(self, weights_path):
        """Load fine-tuned weights, plus a sibling .mapping.yaml if present."""
        self.model = _model_class(self.model_size)(pretrain_weights=str(weights_path), device=self.device)
        self._load_sibling_mapping(weights_path)
