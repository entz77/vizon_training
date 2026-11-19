"""
Image preprocessor.

Runs a chain of preprocessing steps on an image or a folder of images. Steps:
  - perspective: detect boxes with the selected model, fit regression lines through the box
    anchors (bottom / center / top) of each row, find the side the rows converge to and stretch
    that side so the rows become parallel. Rows whose slope is an outlier against the other rows
    are dropped first (see src/preprocessing/perspective.py).
  - crop: if the image was warped, run the same model again on the corrected image; then crop
    it vertically to the topmost box's y1 and the bottommost box's y2 (plus --crop-buffer px),
    cutting away the black "fan" wedges the warp leaves above and below the shelf.

Outputs, under --output:
  <name>.<ext>              the preprocessed image (sub-folders of --source preserved)
  debug/<name>.<ext>        detections, row lines and verdict drawn on the original (--debug)
  debug/<name>_crop.<ext>   detections on the corrected image and the crop limits (--debug)
  preprocess.json           per-image record: detections, analysis, homography, rectified boxes,
                            crop range and the detections in output-image coordinates
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

from src.inference import DetectorPredictor
from src.models import available_models, build_model
from src.preprocessing import (ANCHORS, analyze_perspective, draw_analysis, rectify_image,
                               transform_boxes)

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff'}
STEPS = ('perspective', 'crop')


def _boxes(predictions):
    return [[d['bbox'][k] for k in ('x1', 'y1', 'x2', 'y2')] for d in predictions['detections']]


def perspective_step(image, predictions, args, record, debug_path=None):
    boxes = _boxes(predictions)
    # Only fit rows on the classes that stand on a common baseline; excluded boxes are still
    # drawn and transformed, and line.box_indices keep pointing into the full detection list.
    excluded = set(args.exclude_classes or ())
    keep = [i for i, d in enumerate(predictions['detections'])
            if d.get('class_name') not in excluded]
    analysis = analyze_perspective(
        [boxes[i] for i in keep], image.shape[:2],
        anchor=args.anchor,
        min_boxes_per_row=args.min_boxes_per_row,
        row_tolerance=args.row_tolerance,
        min_stretch_ratio=args.min_stretch_ratio,
        max_stretch_ratio=args.max_stretch_ratio,
        max_row_slope=args.max_row_slope,
        slope_tolerance=args.slope_tolerance,
    )
    for line in analysis.lines + analysis.rejected_lines:
        line.box_indices = [keep[i] for i in line.box_indices]
    if debug_path is not None:
        debug_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(debug_path), draw_analysis(image, boxes, analysis))

    apply = analysis.homography is not None and analysis.converges_to in ('left', 'right')
    record['perspective'] = {**analysis.to_dict(), 'applied': apply}
    if not apply:
        return image
    record['perspective']['rectified_boxes'] = transform_boxes(boxes, analysis.homography).tolist()
    return rectify_image(image, analysis)


def crop_step(image, predictions, args, record, reinferred, debug_path=None):
    """Crop to [topmost y1 - buffer, bottommost y2 + buffer], clamped to the image."""
    h = image.shape[0]
    boxes = _boxes(predictions)
    info = {'reinferred': reinferred, 'buffer': args.crop_buffer, 'applied': bool(boxes)}
    record['crop'] = info
    if not boxes:
        info['reason'] = 'no detections'
        return image
    y0 = max(0, int(min(b[1] for b in boxes)) - args.crop_buffer)
    y1 = min(h, int(max(b[3] for b in boxes) + 0.999) + args.crop_buffer)
    info['y_range'] = [y0, y1]
    # detections in output-image coordinates
    info['detections'] = [
        {**d, 'bbox': {**d['bbox'], 'y1': d['bbox']['y1'] - y0, 'y2': d['bbox']['y2'] - y0}}
        for d in predictions['detections']
    ]
    if debug_path is not None:
        dbg = DetectorPredictor._draw_boxes(image.copy(), predictions)
        w = image.shape[1]
        for y in (y0, y1):
            cv2.line(dbg, (0, y), (w, y), (0, 0, 255), 3)
        debug_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(debug_path.with_name(f'{debug_path.stem}_crop{debug_path.suffix}')), dbg)
    return image[y0:y1]


def main(args):
    source = Path(args.source)
    images = ([source] if source.is_file()
              else sorted(p for p in source.rglob('*') if p.suffix.lower() in IMAGE_EXTENSIONS))
    if not images:
        raise SystemExit(f"No images found in {source}")
    root = source.parent if source.is_file() else source
    out_dir = Path(args.output)

    model = build_model(args.model_type, args.model_size, device=args.device)
    model.load_weights(args.weights)
    if args.class_id_map:
        model.load_class_id_map(args.class_id_map)
    predictor = DetectorPredictor(model, conf_threshold=args.conf, iou_threshold=args.iou)

    records = []
    for i in tqdm(range(0, len(images), args.batch_size), desc="Preprocessing", unit="batch"):
        chunk = images[i:i + args.batch_size]
        preds = predictor.predict_batch([str(p) for p in chunk], verbose=False)
        items = []  # (rel, image, pred, record, warped)
        for path, pred in zip(chunk, preds):
            rel = path.relative_to(root)
            image = cv2.imread(str(path))
            if image is None:
                print(f"Could not read {path}, skipping")
                continue
            record = {'image_path': str(path), 'detections': pred['detections']}
            warped = False
            if 'perspective' in args.steps:
                debug = out_dir / 'debug' / rel if args.debug else None
                image = perspective_step(image, pred, args, record, debug)
                warped = record['perspective']['applied']
            items.append([rel, image, pred, record, warped])

        if 'crop' in args.steps:
            # Detect again on the warped images (one batch) so the crop uses boxes that match them.
            redo = [it for it in items if it[4]]
            if redo:
                for it, new in zip(redo, predictor.predict_batch([it[1] for it in redo], verbose=False)):
                    it[2] = new
            for it in items:
                rel, image, pred, record, warped = it
                debug = out_dir / 'debug' / rel if args.debug else None
                it[1] = crop_step(image, pred, args, record, warped, debug)

        for rel, image, _, record, _ in items:
            out_path = out_dir / rel
            out_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out_path), image)
            record['output_path'] = str(out_path)
            records.append(record)

    with open(out_dir / 'preprocess.json', 'w') as f:
        json.dump(records, f, indent=2)

    if 'perspective' in args.steps:
        verdicts = {}
        for r in records:
            p = r['perspective']
            key = p['converges_to'] if p['applied'] else 'not corrected'
            verdicts[key] = verdicts.get(key, 0) + 1
        print("Perspective:", ", ".join(f"{k}: {v}" for k, v in sorted(verdicts.items())))
    if 'crop' in args.steps:
        print(f"Cropped: {sum(r['crop']['applied'] for r in records)}/{len(records)}")
    print(f"Wrote {len(records)} image(s) and preprocess.json to {out_dir}")


if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Preprocess images (perspective rectification, ...)')
    p.add_argument('--source', required=True, help='Image file or folder (searched recursively)')
    p.add_argument('--output', required=True, help='Output folder')
    p.add_argument('--steps', nargs='+', default=list(STEPS), choices=STEPS,
                   help='Preprocessing steps to run, in order')
    p.add_argument('--debug', action='store_true',
                   help='Also save debug images with rows and regression lines drawn')

    m = p.add_argument_group('model')
    m.add_argument('--model-type', default='yolo', choices=available_models())
    m.add_argument('--model-size', default=None)
    m.add_argument('--weights', required=True)
    m.add_argument('--class-id-map', default=None)
    m.add_argument('--conf', type=float, default=0.5)
    m.add_argument('--iou', type=float, default=0.45)
    m.add_argument('--device', default='cpu', choices=['cuda', 'cpu'])
    m.add_argument('--batch-size', type=int, default=8)

    g = p.add_argument_group('perspective')
    g.add_argument('--anchor', default='bottom', choices=ANCHORS,
                   help='Box point the row regression lines are fitted through')
    g.add_argument('--min-boxes-per-row', type=int, default=3,
                   help='Rows with fewer boxes are ignored')
    g.add_argument('--row-tolerance', type=float, default=0.5,
                   help='Max distance from a row line to join that row, in median box heights')
    g.add_argument('--min-stretch-ratio', type=float, default=1.03,
                   help='Below this near/far spread ratio the image is left as is')
    g.add_argument('--max-stretch-ratio', type=float, default=3.0,
                   help='Above this ratio the fit is distrusted and the image is left as is')
    g.add_argument('--max-row-slope', type=float, default=0.5,
                   help='Rows steeper than this |dy/dx| are dropped as mis-grouped')
    g.add_argument('--slope-tolerance', type=float, default=0.03,
                   help='Rows whose slope deviates more than this from the slope-vs-height trend '
                        'of the other rows are dropped as outliers')
    g.add_argument('--exclude-classes', nargs='*', default=[],
                   help='Class names left out of the row fit (e.g. Shelf)')

    c = p.add_argument_group('crop')
    c.add_argument('--crop-buffer', type=int, default=10,
                   help='Pixels kept above the topmost and below the bottommost box '
                        '(clamped to the image)')

    main(p.parse_args())
