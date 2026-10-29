"""
Backend-independent evaluation report (works with any BaseDetector).

Runs the model over a dataset split once and writes, into `out_dir`:
    report.json                      summary metrics, per-class P/R/F1/AP, confusion matrix
    samples/                         ground truth | predictions side by side, plus samples_grid.jpg
    confusion_matrix.png             counts, rows = predicted, columns = ground truth
    confusion_matrix_normalized.png  columns normalised to ground-truth totals
    PR_curve.png / P_curve.png / R_curve.png / F1_curve.png

Predicted class IDs are compared with the raw label IDs (class_id_map is not applied).
"""

import json
from pathlib import Path

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from .split_metrics import class_names, iter_split

CONF_GRID = np.linspace(0, 1, 101)


def _iou(a, b):
    """Pairwise IoU between xyxy boxes a[N,4] and b[M,4] -> [N,M]."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    ix1 = np.maximum(a[:, None, 0], b[None, :, 0])
    iy1 = np.maximum(a[:, None, 1], b[None, :, 1])
    ix2 = np.minimum(a[:, None, 2], b[None, :, 2])
    iy2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)


def _confusion(images, nc, conf, iou_thr):
    """(nc+1)x(nc+1) matrix, rows = predicted, cols = ground truth, last index = background."""
    cm = np.zeros((nc + 1, nc + 1), dtype=np.int64)
    for im in images:
        keep = im['scores'] >= conf
        boxes, labels = im['boxes'][keep], im['labels'][keep]
        gt_boxes, gt_labels = im['gt_boxes'], im['gt_labels']
        ious = _iou(boxes, gt_boxes)
        pairs = [(ious[i, j], i, j) for i, j in zip(*np.where(ious >= iou_thr))]
        pairs.sort(reverse=True)
        used_p, used_g = set(), set()
        for _, i, j in pairs:
            if i in used_p or j in used_g:
                continue
            used_p.add(i)
            used_g.add(j)
            cm[labels[i], gt_labels[j]] += 1
        for i in range(len(boxes)):
            if i not in used_p:
                cm[labels[i], nc] += 1
        for j in range(len(gt_boxes)):
            if j not in used_g:
                cm[nc, gt_labels[j]] += 1
    return cm


def _class_matches(images, c, iou_thr):
    """Scores and TP flags of every class-`c` prediction (greedy by score), plus #GT of class c."""
    scores, tps, n_gt = [], [], 0
    for im in images:
        gt = im['gt_boxes'][im['gt_labels'] == c]
        n_gt += len(gt)
        sel = im['labels'] == c
        boxes, sc = im['boxes'][sel], im['scores'][sel]
        order = np.argsort(-sc)
        boxes, sc = boxes[order], sc[order]
        ious = _iou(boxes, gt)
        taken = set()
        for i in range(len(boxes)):
            tp = False
            if ious.shape[1]:
                j = int(np.argmax([-1 if k in taken else ious[i, k] for k in range(ious.shape[1])]))
                if j not in taken and ious[i, j] >= iou_thr:
                    taken.add(j)
                    tp = True
            scores.append(sc[i])
            tps.append(tp)
    return np.array(scores), np.array(tps, dtype=bool), n_gt


def _pr_curves(scores, tps, n_gt):
    """Precision/recall/F1 over the confidence grid, plus PR curve points and AP (101-point)."""
    p = np.zeros(len(CONF_GRID))
    r = np.zeros(len(CONF_GRID))
    for k, t in enumerate(CONF_GRID):
        sel = scores >= t
        tp = int(tps[sel].sum())
        n = int(sel.sum())
        p[k] = tp / n if n else 0.0
        r[k] = tp / n_gt if n_gt else 0.0
    f1 = 2 * p * r / (p + r + 1e-9)

    if n_gt == 0 or len(scores) == 0:
        return p, r, f1, np.array([0.0, 1.0]), np.array([0.0, 0.0]), 0.0
    order = np.argsort(-scores)
    ctp = np.cumsum(tps[order])
    cfp = np.cumsum(~tps[order])
    rec = ctp / n_gt
    prec = ctp / (ctp + cfp)
    # monotone precision envelope, sampled at 101 recall points (COCO-style)
    env = np.maximum.accumulate(prec[::-1])[::-1]
    grid = np.linspace(0, 1, 101)
    idx = np.searchsorted(rec, grid, side='left')
    ap = float(np.mean([env[i] if i < len(env) else 0.0 for i in idx]))
    return p, r, f1, rec, env, ap


def _plot_confusion(cm, names, path, normalize):
    labels = list(names) + ['background']
    data = cm.astype(np.float64)
    if normalize:
        data = data / np.maximum(data.sum(axis=0, keepdims=True), 1)
    n = len(labels)
    fig, ax = plt.subplots(figsize=(max(6, n * 0.9), max(5, n * 0.8)))
    im = ax.imshow(data, cmap='Blues', vmin=0, vmax=1 if normalize else None)
    ax.set_xticks(range(n), labels, rotation=45, ha='right')
    ax.set_yticks(range(n), labels)
    ax.set_xlabel('Ground truth')
    ax.set_ylabel('Predicted')
    ax.set_title('Confusion matrix' + (' (normalized)' if normalize else ''))
    thresh = data.max() / 2 if data.max() else 0
    for i in range(n):
        for j in range(n):
            txt = f'{data[i, j]:.2f}' if normalize else f'{int(data[i, j])}'
            ax.text(j, i, txt, ha='center', va='center',
                    color='white' if data[i, j] > thresh else 'black', fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_curves(x, ys, names, xlabel, ylabel, title, path, mean_label=None, xlim=(0, 1)):
    fig, ax = plt.subplots(figsize=(7, 5))
    xs_list = x if isinstance(x, list) else [x] * len(ys)
    for xs, y, name in zip(xs_list, ys, names):
        ax.plot(xs, y, linewidth=1, label=name)
    if mean_label is not None:
        mx = x[0] if isinstance(x, list) else x
        ax.plot(mx, np.mean(ys, axis=0), color='black', linewidth=3, label=mean_label)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xlim(*xlim)
    ax.set_ylim(0, 1.05)
    ax.legend(loc='lower left', fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _color(c):
    """Stable, distinct BGR color per class id."""
    hue = int((c * 47) % 180)
    bgr = cv2.cvtColor(np.uint8([[[hue, 220, 255]]]), cv2.COLOR_HSV2BGR)[0, 0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


def _draw(img, boxes, labels, names, scores=None):
    img = img.copy()
    thick = max(2, round(max(img.shape[:2]) / 400))
    scale = max(0.5, max(img.shape[:2]) / 1200)
    for i, (x1, y1, x2, y2) in enumerate(boxes):
        c = int(labels[i])
        color = _color(c)
        text = names[c] if 0 <= c < len(names) else str(c)
        if scores is not None:
            text += f' {scores[i]:.2f}'
        p1, p2 = (int(x1), int(y1)), (int(x2), int(y2))
        cv2.rectangle(img, p1, p2, color, thick)
        (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
        top = max(p1[1] - th - base, 0)
        cv2.rectangle(img, (p1[0], top), (p1[0] + tw, top + th + base), color, -1)
        cv2.putText(img, text, (p1[0], top + th), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 1, cv2.LINE_AA)
    return img


def _label(img, text):
    """Attach a title banner above the image (so it never covers boxes)."""
    scale = max(0.6, max(img.shape[:2]) / 1200)
    banner = np.full((int(44 * scale), img.shape[1], 3), 32, dtype=np.uint8)
    cv2.putText(banner, text, (8, int(30 * scale)), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 1, cv2.LINE_AA)
    return np.vstack([banner, img])


def _save_samples(images, names, conf, out_dir, num_samples):
    """Ground truth | predictions side by side for evenly spaced images, plus a grid of them."""
    if num_samples <= 0 or not images:
        return
    samples_dir = out_dir / 'samples'
    samples_dir.mkdir(exist_ok=True)
    idx = np.unique(np.linspace(0, len(images) - 1, min(num_samples, len(images))).astype(int))

    pairs = []
    for k in idx:
        im = images[k]
        img = cv2.imread(im['path'])
        if img is None:
            continue
        keep = im['scores'] >= conf
        gt = _label(_draw(img, im['gt_boxes'], im['gt_labels'], names), 'ground truth')
        pred = _label(_draw(img, im['boxes'][keep], im['labels'][keep], names, im['scores'][keep]),
                      f'predictions (conf>={conf})')
        pair = np.hstack([gt, pred])
        cv2.imwrite(str(samples_dir / f'{Path(im["path"]).stem}.jpg'), pair)
        pairs.append(pair)

    if pairs:
        # grid: 2 pairs per row, each scaled to a common width
        w = 1000
        tiles = [cv2.resize(p, (w, int(p.shape[0] * w / p.shape[1]))) for p in pairs]
        rows = []
        for i in range(0, len(tiles), 2):
            row = tiles[i:i + 2]
            if len(row) == 1:
                row.append(np.zeros_like(row[0]))
            h = max(t.shape[0] for t in row)
            row = [cv2.copyMakeBorder(t, 0, h - t.shape[0], 0, 0, cv2.BORDER_CONSTANT) for t in row]
            rows.append(np.hstack(row))
        cv2.imwrite(str(out_dir / 'samples_grid.jpg'), np.vstack(rows))


def build_report(model, data_yaml, split='val', out_dir='report', conf=0.25, iou=0.5, batch_size=8, num_samples=16):
    """
    Evaluate `model` on one split and write the report files (see module docstring).

    Args:
        model: BaseDetector
        data_yaml: dataset config (path, <split>, names)
        split: 'train' | 'val' | 'test'
        out_dir: directory to write the report into
        conf: confidence threshold for the confusion matrix and the per-class P/R/F1 table
        iou: IoU threshold for matching predictions to ground truth (P/R/F1/AP50/confusion)
        batch_size: images per predict() call
        num_samples: number of evenly spaced images to draw ground truth vs predictions for (0 disables)

    Returns:
        dict: the contents of report.json
    """
    from torchmetrics.detection import MeanAveragePrecision

    names = class_names(data_yaml)
    nc = len(names)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    images, out_of_range = [], 0
    coco = MeanAveragePrecision(box_format='xyxy', iou_type='bbox')
    # low floor so the curves and mAP see the whole score range
    for path, dets, gt_boxes, gt_labels in iter_split(model, data_yaml, split, conf=0.001, batch_size=batch_size):
        boxes = np.array([d.xyxy for d in dets], dtype=np.float32).reshape(-1, 4)
        scores = np.array([d.confidence for d in dets], dtype=np.float32)
        labels = np.array([d.class_id for d in dets], dtype=np.int64)
        valid = (labels >= 0) & (labels < nc)
        out_of_range += int((~valid).sum())
        images.append({
            'path': str(path), 'boxes': boxes[valid], 'scores': scores[valid], 'labels': labels[valid],
            'gt_boxes': gt_boxes, 'gt_labels': gt_labels,
        })
        coco.update(
            [{'boxes': torch.from_numpy(boxes[valid]), 'scores': torch.from_numpy(scores[valid]),
              'labels': torch.from_numpy(labels[valid])}],
            [{'boxes': torch.from_numpy(gt_boxes), 'labels': torch.from_numpy(gt_labels)}],
        )

    cm = _confusion(images, nc, conf, iou)

    per_class, curves = [], []
    for c in range(nc):
        scores, tps, n_gt = _class_matches(images, c, iou)
        p, r, f1, rec, env, ap = _pr_curves(scores, tps, n_gt)
        curves.append((p, r, f1, rec, env))
        k = int(round(conf * (len(CONF_GRID) - 1)))
        per_class.append({
            'class_id': c, 'name': names[c], 'support': int(n_gt),
            'precision': float(p[k]), 'recall': float(r[k]), 'f1': float(f1[k]),
            'ap50' if iou == 0.5 else f'ap@{iou}': ap,
            'best_f1': float(f1.max()), 'best_f1_conf': float(CONF_GRID[int(f1.argmax())]),
        })

    with_gt = [i for i, c in enumerate(per_class) if c['support'] > 0]
    mean_f1 = np.mean([curves[i][2] for i in with_gt], axis=0) if with_gt else np.zeros(len(CONF_GRID))
    coco_out = coco.compute()
    report = {
        'split': split, 'num_images': len(images), 'conf': conf, 'iou': iou,
        'map50_95': float(coco_out['map']), 'map50': float(coco_out['map_50']),
        'map75': float(coco_out['map_75']),
        'best_f1': float(mean_f1.max()), 'best_f1_conf': float(CONF_GRID[int(mean_f1.argmax())]),
        'per_class': per_class,
        'confusion_matrix': {'labels': names + ['background'], 'rows': 'predicted', 'cols': 'ground truth',
                             'matrix': cm.tolist()},
        'warnings': [],
    }
    if out_of_range:
        report['warnings'].append(
            f'{out_of_range} predictions had class IDs outside [0, {nc}) and were ignored; '
            'check the model/dataset class ID convention'
        )

    _plot_confusion(cm, names, out_dir / 'confusion_matrix.png', normalize=False)
    _plot_confusion(cm, names, out_dir / 'confusion_matrix_normalized.png', normalize=True)
    used = [names[i] for i in with_gt]
    if used:
        pick = lambda k: [curves[i][k] for i in with_gt]  # noqa: E731
        _plot_curves([c for c in pick(3)], pick(4), used, 'Recall', 'Precision',
                     f'Precision-Recall @ IoU {iou}', out_dir / 'PR_curve.png')
        for key, k, label in (('P', 0, 'Precision'), ('R', 1, 'Recall'), ('F1', 2, 'F1')):
            _plot_curves(CONF_GRID, pick(k), used, 'Confidence', label, f'{label}-Confidence',
                         out_dir / f'{key}_curve.png', mean_label='all classes')

    _save_samples(images, names, conf, out_dir, num_samples)

    with open(out_dir / 'report.json', 'w') as f:
        json.dump(report, f, indent=2)
    return report
