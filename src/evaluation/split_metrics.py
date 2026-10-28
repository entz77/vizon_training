"""
Backend-independent mAP evaluation of a detector on a YOLO-format dataset split.

Runs model.predict() over the split's images and scores the result against the
YOLO txt labels, so any BaseDetector can be validated the same way.
"""

from pathlib import Path

import numpy as np
import torch
import yaml

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff'}


def _label_path(img_path: Path, img_dir: Path, label_dir: Path) -> Path:
    return label_dir / img_path.relative_to(img_dir).with_suffix('.txt')


def _read_gt(label_file: Path, width: int, height: int):
    """YOLO normalized xywh -> absolute xyxy. Returns (boxes[N,4], labels[N])."""
    if not label_file.exists():
        return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.int64)
    rows = [line.split() for line in label_file.read_text().splitlines() if line.strip()]
    if not rows:
        return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.int64)
    arr = np.array([[float(v) for v in r[:5]] for r in rows], dtype=np.float32)
    cx, cy, w, h = arr[:, 1] * width, arr[:, 2] * height, arr[:, 3] * width, arr[:, 4] * height
    boxes = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)
    return boxes, arr[:, 0].astype(np.int64)


def evaluate_split(model, data_yaml, split='val', conf=0.001, batch_size=8):
    """
    Compute COCO-style mAP of `model` on one split of a YOLO dataset config.

    Args:
        model: BaseDetector
        data_yaml: dataset config (path, <split>: images dir, names)
        split: 'train' | 'val' | 'test'
        conf: confidence floor for predictions (keep low for mAP)
        batch_size: images held in memory per predict() call

    Returns:
        dict with map, map50, map75, per-class AP under 'per_class_map', and 'num_images'
    """
    from torchmetrics.detection import MeanAveragePrecision

    with open(data_yaml, 'r') as f:
        cfg = yaml.safe_load(f)
    if not cfg.get(split):
        raise ValueError(f"'{split}' split is not defined in {data_yaml}")

    root = Path(cfg.get('path', Path(data_yaml).parent)).resolve()
    img_dir = (root / cfg[split]).resolve()
    # Ultralytics convention: .../images/<split> pairs with .../labels/<split>
    label_dir = Path(str(img_dir).replace('/images/', '/labels/'))
    images = sorted(p for p in img_dir.rglob('*') if p.suffix.lower() in IMAGE_EXTENSIONS)
    if not images:
        raise ValueError(f"No images found in {img_dir}")

    metric = MeanAveragePrecision(box_format='xyxy', iou_type='bbox', class_metrics=True)

    for i in range(0, len(images), batch_size):
        chunk = images[i:i + batch_size]
        results = model.predict([str(p) for p in chunk], conf=conf, verbose=False)
        preds, targets = [], []
        for path, res in zip(chunk, results):
            height, width = res.image_shape
            gt_boxes, gt_labels = _read_gt(_label_path(path, img_dir, label_dir), width, height)
            dets = res.detections
            preds.append({
                'boxes': torch.tensor([d.xyxy for d in dets], dtype=torch.float32).reshape(-1, 4),
                'scores': torch.tensor([d.confidence for d in dets], dtype=torch.float32),
                'labels': torch.tensor([d.class_id for d in dets], dtype=torch.int64),
            })
            targets.append({
                'boxes': torch.from_numpy(gt_boxes),
                'labels': torch.from_numpy(gt_labels),
            })
        metric.update(preds, targets)

    out = metric.compute()
    classes = out['classes'].reshape(-1).tolist()
    per_class = out['map_per_class'].reshape(-1).tolist()
    return {
        'map': float(out['map']),
        'map50': float(out['map_50']),
        'map75': float(out['map_75']),
        # torchmetrics reports -1 for classes with no ground truth in the split
        'per_class_map': {int(c): float(v) for c, v in zip(classes, per_class) if v >= 0},
        'num_images': len(images),
    }
