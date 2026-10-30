"""
Compare RF-DETR inference with one model call per image (current default) against one call per
batch of images (RFDETRModel(batched=True)): speed, peak GPU memory, and whether the detections match.

Example (on the GPU machine):
    python benchmark_rfdetr_batching.py --weights models/rfdetr_base_v7.pth --model-size base \\
        --images path/to/images/ --batch-sizes 1 4 8 16 --conf 0.25

Add --fp16 to also time float16 inference (model.inference(dtype=float16); CUDA only).

Weights are optional: without --weights the pretrained COCO checkpoint is used, which is enough
to check speed and loop-vs-batch equality (class names will just be COCO's).
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))

from src.models import build_model

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff'}


def sync(device):
    if device == 'cuda':
        torch.cuda.synchronize()


def run(model, paths, batch_size, batched, conf, device):
    """Predict every path, `batch_size` per call. Returns (seconds, peak_mem_mb, results)."""
    if device == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    results = []
    sync(device)
    start = time.perf_counter()
    for i in range(0, len(paths), batch_size):
        results += model.predict([str(p) for p in paths[i:i + batch_size]], conf=conf,
                                 verbose=False, batched=batched)
    sync(device)
    elapsed = time.perf_counter() - start
    peak = torch.cuda.max_memory_allocated() / 2**20 if device == 'cuda' else float('nan')
    return elapsed, peak, results


def compare(loop_results, batch_results):
    """How far the batched detections drift from the per-image ones."""
    count_diff = 0
    class_diff = 0
    max_box = 0.0
    max_score = 0.0
    for a, b in zip(loop_results, batch_results):
        if len(a.detections) != len(b.detections):
            count_diff += 1
            continue
        # order by score so tied/near-tied detections line up
        da = sorted(a.detections, key=lambda d: -d.confidence)
        db = sorted(b.detections, key=lambda d: -d.confidence)
        for x, y in zip(da, db):
            class_diff += x.class_id != y.class_id
            max_box = max(max_box, float(np.abs(np.array(x.xyxy) - np.array(y.xyxy)).max()))
            max_score = max(max_score, abs(x.confidence - y.confidence))
    return {'images_with_different_detection_count': count_diff, 'class_mismatches': int(class_diff),
            'max_box_diff_px': max_box, 'max_score_diff': max_score}


def main(args):
    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    paths = sorted(p for p in Path(args.images).rglob('*') if p.suffix.lower() in IMAGE_EXTENSIONS)
    if args.limit:
        paths = paths[:args.limit]
    if not paths:
        raise SystemExit(f'No images found in {args.images}')

    print(f'Device: {device}' + (f' ({torch.cuda.get_device_name(0)})' if device == 'cuda' else ''))
    print(f'Images: {len(paths)} from {args.images}')
    model = build_model('rfdetr', args.model_size, device=device)
    if args.weights:
        model.load_weights(args.weights)

    precisions = ['fp32'] + (['fp16'] if args.fp16 and device == 'cuda' else [])
    if args.fp16 and device != 'cuda':
        print('--fp16 ignored: needs a CUDA device')

    rows, ref = [], {}
    print(f"\n{'prec':>4} {'batch':>5} {'mode':>8} {'img/s':>8} {'sec':>8} {'peak MB':>9}  vs fp32 per-image loop")
    for prec in precisions:
        if prec == 'fp16':
            model.optimize_for_inference(dtype='float16', compile=False)
        print(f'Warming up ({prec})...')
        for batched in (False, True):
            run(model, paths[:args.warmup], args.warmup, batched, args.conf, device)

        for bs in args.batch_sizes:
            for batched in (False, True):
                best = None
                for _ in range(args.repeats):
                    t, mem, res = run(model, paths, bs, batched, args.conf, device)
                    best = (t, mem, res) if best is None or t < best[0] else best
                t, mem, res = best
                if prec == 'fp32' and not batched:
                    ref[bs] = res
                diff = None if prec == 'fp32' and not batched else compare(ref[bs], res)
                mode = 'batched' if batched else 'loop'
                rows.append({'precision': prec, 'batch_size': bs, 'mode': mode, 'seconds': t,
                             'img_per_s': len(paths) / t, 'peak_mem_mb': mem, 'vs_fp32_loop': diff})
                note = '(reference)' if diff is None else (
                    f"counts differ in {diff['images_with_different_detection_count']}, "
                    f"class mismatches {diff['class_mismatches']}, box {diff['max_box_diff_px']:.2f}px, "
                    f"score {diff['max_score_diff']:.4f}")
                print(f"{prec:>4} {bs:>5} {mode:>8} {len(paths) / t:>8.2f} {t:>8.2f} {mem:>9.0f}  {note}")

    if args.output:
        Path(args.output).write_text(json.dumps(
            {'device': device, 'num_images': len(paths), 'rows': rows}, indent=2))
        print(f'\nSaved: {args.output}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Benchmark RF-DETR per-image vs batched inference')
    parser.add_argument('--images', required=True, help='Folder of images (recursive)')
    parser.add_argument('--weights', default=None, help='Fine-tuned checkpoint (default: pretrained COCO)')
    parser.add_argument('--model-size', default='base')
    parser.add_argument('--batch-sizes', type=int, nargs='+', default=[1, 4, 8, 16])
    parser.add_argument('--conf', type=float, default=0.25)
    parser.add_argument('--limit', type=int, default=0, help='Use only the first N images')
    parser.add_argument('--warmup', type=int, default=4, help='Images used for warm-up per mode')
    parser.add_argument('--repeats', type=int, default=2, help='Timed runs per setting (best is kept)')
    parser.add_argument('--fp16', action='store_true', help='Also benchmark float16 (CUDA only)')
    parser.add_argument('--device', default=None, choices=['cuda', 'cpu'])
    parser.add_argument('--output', default=None, help='Write results as JSON')
    main(parser.parse_args())
