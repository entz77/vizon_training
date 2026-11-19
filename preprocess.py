"""
Image preprocessor.

Runs a chain of preprocessing steps on an image or a folder of images. Steps:
  - perspective: detect boxes with the selected model, fit regression lines through the box
    anchors (bottom / center / top) of each row, find the side the rows converge to and stretch
    that side so the rows become parallel. Rows whose slope is an outlier against the other rows
    are dropped first (see src/preprocessing/perspective.py).
  - crop: only for images the perspective step warped - run the same model again on the
    corrected image, then crop it vertically to the topmost box's y1 and the bottommost box's y2
    (plus --crop-buffer px), cutting away the black "fan" wedges the warp leaves above and below
    the shelf. Images that weren't warped are left uncropped.

With --horizontal, a second pass then runs on the result of the first, using only the shelf
boxes (--h-shelf-classes). Trimmed edge shelves (cut off by the left / right border and not
reaching the image centre) are removed first; then a line through the shelves' left ends and one
through their right ends (ends cut off by the border are skipped) give the shelf unit's sides; if they converge
towards the top or bottom, that edge is stretched horizontally so the sides become parallel
(analyze_shelf_edges in src/preprocessing/perspective.py). As in the first pass, a corrected
image is detected again and cropped, here to the leftmost box's x1 and the rightmost box's x2.

Outputs, under --output:
  <name>.<ext>                the preprocessed image (sub-folders of --source preserved)
  debug/<name>.<ext>          detections, row lines and verdict drawn on the original (--debug)
  debug/<name>_crop.<ext>     detections on the corrected image and the crop limits (--debug)
  debug/<name>_h.<ext>        same as the two above for the horizontal pass (--debug,
  debug/<name>_h_crop.<ext>   --horizontal)
  preprocess.json             per-image record: detections, per pass the analysis, homography,
                              rectified boxes, crop range and detections after the crop, and
                              output_detections - the detections in output-image coordinates
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
from src.preprocessing import (ANCHORS, analyze_perspective, analyze_shelf_edges, draw_analysis,
                               rectify_image, transform_boxes)

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff'}
STEPS = ('perspective', 'crop')


def _boxes(detections):
    return [[d['bbox'][k] for k in ('x1', 'y1', 'x2', 'y2')] for d in detections]


# Per pass: key suffix in the record / debug file names, and the image axis the crop works
# along (1 = y / rows, 0 = x / columns of the xyxy box).
PASSES = {
    'vertical': dict(suffix='', axis=1),
    'horizontal': dict(suffix='_h', axis=0),
}


def _debug_path(args, out_dir, rel, suffix):
    return out_dir / 'debug' / rel.with_name(f'{rel.stem}{suffix}{rel.suffix}') if args.debug else None


def perspective_step(image, detections, args, record, direction, debug_path=None):
    """Analyse one direction and warp the image if the analysis says so. Returns (image, warped)."""
    cfg = PASSES[direction]
    boxes = _boxes(detections)
    # Only use the relevant classes; the other boxes are still drawn and transformed, and
    # line.box_indices keep pointing into the full detection list.
    if direction == 'vertical':
        excluded = set(args.exclude_classes or ())
        keep = [i for i, d in enumerate(detections) if d.get('class_name') not in excluded]
    else:
        shelves = set(args.h_shelf_classes)
        keep = [i for i, d in enumerate(detections) if d.get('class_name') in shelves]
    kept_boxes = [boxes[i] for i in keep]
    if direction == 'vertical':
        analysis = analyze_perspective(
            kept_boxes, image.shape[:2],
            anchor=args.anchor,
            min_boxes_per_row=args.min_boxes_per_row,
            row_tolerance=args.row_tolerance,
            min_stretch_ratio=args.min_stretch_ratio,
            max_stretch_ratio=args.max_stretch_ratio,
            max_row_slope=args.max_row_slope,
            slope_tolerance=args.slope_tolerance,
        )
    else:
        analysis = analyze_shelf_edges(
            kept_boxes, image.shape[:2],
            edge_margin=args.h_edge_margin,
            tolerance=args.h_tolerance,
            min_points=args.h_min_points,
            min_stretch_ratio=args.min_stretch_ratio,
            max_stretch_ratio=args.h_max_stretch_ratio,
        )
    for line in analysis.lines + analysis.rejected_lines:
        line.box_indices = [keep[i] for i in line.box_indices]
    analysis.rejected_boxes = [keep[i] for i in analysis.rejected_boxes]
    if debug_path is not None:
        debug_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(debug_path), draw_analysis(image, boxes, analysis))

    apply = analysis.homography is not None and analysis.converges_to not in (None, 'none')
    rec = record['perspective' + cfg['suffix']] = {**analysis.to_dict(), 'applied': apply}
    if not apply:
        return image, False
    rec['rectified_boxes'] = transform_boxes(boxes, analysis.homography).tolist()
    return rectify_image(image, analysis), True


def crop_step(image, detections, args, record, direction, debug_path=None):
    """
    Crop to the box extent along the pass' axis, plus --crop-buffer, clamped to the image:
    vertical -> [topmost y1, bottommost y2], horizontal -> [leftmost x1, rightmost x2].
    Returns (image, detections in the cropped image's coordinates).
    """
    cfg = PASSES[direction]
    axis = cfg['axis']
    lo_key, hi_key = ('y1', 'y2') if axis == 1 else ('x1', 'x2')
    size = image.shape[0] if axis == 1 else image.shape[1]
    info = record['crop' + cfg['suffix']] = {'buffer': args.crop_buffer, 'applied': bool(detections)}
    if not detections:
        info['reason'] = 'no detections'
        return image, detections
    lo = max(0, int(min(d['bbox'][lo_key] for d in detections)) - args.crop_buffer)
    hi = min(size, int(max(d['bbox'][hi_key] for d in detections) + 0.999) + args.crop_buffer)
    info['y_range' if axis == 1 else 'x_range'] = [lo, hi]
    shifted = [{**d, 'bbox': {**d['bbox'], lo_key: d['bbox'][lo_key] - lo,
                              hi_key: d['bbox'][hi_key] - lo}} for d in detections]
    info['detections'] = shifted
    if debug_path is not None:
        dbg = DetectorPredictor._draw_boxes(image.copy(), {'detections': detections})
        h, w = image.shape[:2]
        for v in (lo, hi):
            p0, p1 = ((0, v), (w, v)) if axis == 1 else ((v, 0), (v, h))
            cv2.line(dbg, p0, p1, (0, 0, 255), 3)
        debug_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(debug_path), dbg)
    crop = image[lo:hi] if axis == 1 else image[:, lo:hi]
    return crop, shifted


def run_pass(items, direction, predictor, args, out_dir, more_passes):
    """
    One perspective pass over a batch: rectify, then (for the warped images) detect again and
    crop. Each item is a dict with rel, image, detections (in `image` coordinates) and record.
    """
    sfx = PASSES[direction]['suffix']
    warped = []
    for it in items:
        it['image'], ok = perspective_step(it['image'], it['detections'], args, it['record'],
                                           direction, _debug_path(args, out_dir, it['rel'], sfx))
        if ok:
            warped.append(it)
        elif 'crop' in args.steps:
            it['record']['crop' + sfx] = {'applied': False, 'reason': 'no perspective correction'}

    # Detections from before the warp no longer match the image; detect again (one batch) when
    # the crop or a later pass needs them.
    if warped and ('crop' in args.steps or more_passes):
        new = predictor.predict_batch([it['image'] for it in warped], verbose=False)
        for it, pred in zip(warped, new):
            it['detections'] = pred['detections']
            if 'crop' in args.steps:
                it['image'], it['detections'] = crop_step(
                    it['image'], it['detections'], args, it['record'], direction,
                    _debug_path(args, out_dir, it['rel'], sfx + '_crop'))
    elif warped:
        # nothing will use the detections again, but keep output_detections honest
        for it in warped:
            it['detections'] = None


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

    passes = []
    if 'perspective' in args.steps:
        passes = ['vertical'] + (['horizontal'] if args.horizontal else [])

    records = []
    for i in tqdm(range(0, len(images), args.batch_size), desc="Preprocessing", unit="batch"):
        chunk = images[i:i + args.batch_size]
        preds = predictor.predict_batch([str(p) for p in chunk], verbose=False)
        items = []
        for path, pred in zip(chunk, preds):
            image = cv2.imread(str(path))
            if image is None:
                print(f"Could not read {path}, skipping")
                continue
            record = {'image_path': str(path), 'detections': pred['detections']}
            items.append({'rel': path.relative_to(root), 'image': image,
                          'detections': pred['detections'], 'record': record})

        for k, direction in enumerate(passes):
            run_pass(items, direction, predictor, args, out_dir, more_passes=k < len(passes) - 1)

        for it in items:
            record = it['record']
            out_path = out_dir / it['rel']
            out_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out_path), it['image'])
            record['output_path'] = str(out_path)
            record['output_detections'] = it['detections']
            records.append(record)

    with open(out_dir / 'preprocess.json', 'w') as f:
        json.dump(records, f, indent=2)

    for direction in passes:
        sfx = PASSES[direction]['suffix']
        verdicts = {}
        for r in records:
            p = r['perspective' + sfx]
            key = p['converges_to'] if p['applied'] else 'not corrected'
            verdicts[key] = verdicts.get(key, 0) + 1
        line = ", ".join(f"{k}: {v}" for k, v in sorted(verdicts.items()))
        if 'crop' in args.steps:
            line += f" | cropped {sum(r['crop' + sfx]['applied'] for r in records)}/{len(records)}"
        print(f"Perspective ({direction}): {line}")
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

    hz = p.add_argument_group('horizontal perspective (shelf sides)')
    hz.add_argument('--horizontal', action='store_true',
                    help='After the vertical pass, also rectify (and crop) horizontally: make the '
                         'shelf sides, which converge to the top / bottom, parallel')
    hz.add_argument('--h-shelf-classes', nargs='+', default=['Shelf'],
                    help='Class names whose boxes are shelves')
    hz.add_argument('--h-edge-margin', type=float, default=0.01,
                    help='Shelf ends within this fraction of the width of the image border are '
                         'cut off by the frame: a shelf cut off at one border that does not reach '
                         'the image centre is a trimmed edge shelf and left out; otherwise only '
                         'its cut-off end is skipped')
    hz.add_argument('--h-tolerance', type=float, default=0.02,
                    help='Max distance of a shelf end from its side line, as a fraction of the '
                         'width; ends further off are outliers')
    hz.add_argument('--h-min-points', type=int, default=3,
                    help='Min shelf ends on a side to fit that side')
    hz.add_argument('--h-max-stretch-ratio', type=float, default=4.0,
                    help='Above this ratio the fit is distrusted and the image is left as is')

    c = p.add_argument_group('crop')
    c.add_argument('--crop-buffer', type=int, default=10,
                   help='Pixels kept beyond the outermost boxes on each cropped side '
                        '(clamped to the image)')

    main(p.parse_args())
