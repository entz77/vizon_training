"""
Inference script for object detection
"""

import argparse
import json
import sys
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

from src.models import build_model, available_models
from src.inference import DetectorPredictor

# Shared (not per-run, not per-repo) cache directory that GCS-sourced images are downloaded
# into - lives outside this repo, at a fixed absolute path, so it's reachable from
# vizon_classifier_embedding's labelstudio_inference_pipeline.py / lensabonus_inference_pipeline.py
# too (see inference_common.DEFAULT_IMAGES_CACHE_DIR there). All of them preserve the GCS blob's
# own filename locally, so whichever script downloads an image first, the others can reuse it.
DEFAULT_IMAGES_CACHE_DIR = "/home/enitze/projects/repos/vizon/shared_image_folder"


def download_gcs_images(gcs_uris, dest_dir):
    """
    Download a list of gs:// image URIs into dest_dir, skipping any URI whose blob filename
    is already present there (see DEFAULT_IMAGES_CACHE_DIR). URIs that don't exist in GCS
    are skipped with a warning rather than aborting the whole batch.

    Returns:
        list[tuple[str, str]]: (gcs_uri, local_path) pairs for successfully downloaded
        or already-cached images
    """
    from google.api_core.exceptions import NotFound
    from google.cloud import storage

    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    client = storage.Client()
    downloaded = []
    cached = 0
    skipped = 0

    for uri in tqdm(gcs_uris, desc="Downloading", unit="img"):
        bucket_name = uri.split('/')[2]
        blob_name = '/'.join(uri.split('/')[3:])
        local_path = dest_dir / Path(blob_name).name

        if local_path.exists():
            cached += 1
            downloaded.append((uri, str(local_path)))
            continue

        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        try:
            blob.download_to_filename(str(local_path))
        except NotFound:
            skipped += 1
            local_path.unlink(missing_ok=True)
            continue

        downloaded.append((uri, str(local_path)))

    if cached:
        print(f"Reused {cached} already-cached image(s) from {dest_dir}")
    if skipped:
        print(f"Skipped {skipped} image(s) not found in GCS")

    return downloaded


IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff'}


def predict_folder(predictor, source, output, results_json, batch_size):
    """
    Run inference on every image under `source` (recursive), `batch_size` images per model call.

    Writes one JSON with all detections (`results_json`, default: `<output>/results.json` if
    `output` is given, else `<source>_results.json` next to the folder) and, when `output` is
    given, an annotated copy of every image there (sub-folders preserved).
    """
    source = Path(source)
    images = sorted(p for p in source.rglob('*') if p.suffix.lower() in IMAGE_EXTENSIONS)
    if not images:
        raise SystemExit(f"No images found in {source}")
    print(f"Predicting on {len(images)} image(s) in {source} (batch size {batch_size})")

    records = []
    for i in tqdm(range(0, len(images), batch_size), desc="Predicting", unit="batch"):
        chunk = images[i:i + batch_size]
        for path, results in zip(chunk, predictor.predict_batch([str(p) for p in chunk], verbose=False)):
            records.append({**results, 'image_path': str(path)})
            if output:
                predictor.save_annotated(path, results, Path(output) / path.relative_to(source))

    if results_json:
        json_path = Path(results_json)
    elif output:
        json_path = Path(output) / 'results.json'
    else:
        json_path = source.parent / f"{source.name}_results.json"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, 'w') as f:
        json.dump(records, f, indent=2)

    total = sum(len(r['detections']) for r in records)
    print(f"{total} detection(s) in {len(records)} image(s)")
    print(f"Results written to: {json_path}")
    if output:
        print(f"Annotated images saved to: {output}")


def main(args):
    """Main inference function"""
    
    print("=" * 50)
    print("Object Detection Inference")
    print("=" * 50)
    
    # Load model
    print(f"Loading model: {args.weights}")
    model = build_model(args.model_type, args.model_size, device=args.device)
    model.load_weights(args.weights)
    
    # Load class ID mapping if provided
    if args.class_id_map:
        print(f"Loading class ID mapping: {args.class_id_map}")
        model.load_class_id_map(args.class_id_map)
    
    # Initialize predictor
    predictor = DetectorPredictor(
        model=model,
        conf_threshold=args.conf,
        iou_threshold=args.iou
    )
    
    # Run inference based on source type
    if args.mode == 'image' and Path(args.source).is_dir():
        predict_folder(predictor, args.source, args.output, args.results_json, args.batch_size)

    elif args.mode == 'image':
        print(f"Predicting on image: {args.source}")
        results = predictor.predict_image(args.source)
        print(f"Detections: {len(results['detections'])}")
        
        for det in results['detections']:
            print(f"  - Class ID: {det['class_id']}, {det['class_name']}, Confidence: {det['confidence']:.2f}")

        if args.output:
            saved = predictor.save_annotated(args.source, results, args.output)
            print(f"Annotated image saved to: {saved}")
    
    elif args.mode == 'video':
        print(f"Predicting on video: {args.source}")
        results = predictor.predict_video(
            args.source,
            output_path=args.output,
            fps=args.fps
        )
        print(f"Processed {len(results)} frames")
        
        if args.output:
            print(f"Output saved to: {args.output}")
    
    elif args.mode == 'gcs_images':
        with open(args.source, 'r') as f:
            gcs_uris = [line.strip() for line in f if line.strip() and not line.startswith('#')]

        print(f"Downloading {len(gcs_uris)} image(s) from GCS (cache: {args.images_cache_dir})")
        downloaded = download_gcs_images(gcs_uris, args.images_cache_dir)

        output_records = []
        for gcs_uri, local_path in tqdm(downloaded, desc="Predicting", unit="img"):
            results = predictor.predict_image(local_path, verbose=False)
            output_records.append({'gcs_uri': gcs_uri, **results})

        source_path = Path(args.source)
        output_path = args.output or (source_path.parent / f"{source_path.stem}_results.json")
        with open(output_path, 'w') as f:
            json.dump(output_records, f, indent=2)
        print(f"Results written to: {output_path}")

    elif args.mode == 'webcam':
        print(f"Starting webcam inference for {args.duration} seconds")
        results = predictor.predict_webcam(
            duration=args.duration,
            output_path=args.output
        )
        print(f"Processed {len(results)} frames")
        
        if args.output:
            print(f"Output saved to: {args.output}")
    
    print("=" * 50)
    print("Inference completed!")
    print("=" * 50)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Run inference with a detection model'
    )
    
    parser.add_argument(
        '--model-type',
        type=str,
        default='yolo',
        choices=available_models(),
        help='Detection backend'
    )
    parser.add_argument(
        '--model-size',
        type=str,
        default=None,
        help="Model size/variant (backend default if omitted, e.g. 'm' for yolo, 'base' for rfdetr)"
    )
    parser.add_argument(
        '--weights',
        type=str,
        required=True,
        help='Path to model weights'
    )
    parser.add_argument(
        '--source',
        type=str,
        required=True,
        help='Source: image/video path or directory. For --mode gcs_images, a local text file listing gs:// image URIs, one per line'
    )
    parser.add_argument(
        '--mode',
        type=str,
        default='image',
        choices=['image', 'video', 'webcam', 'gcs_images'],
        help='Inference mode'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output path: annotated image (file or folder) in image mode, folder for annotated images when --source is a folder, video in video/webcam mode, JSON in gcs_images mode'
    )
    parser.add_argument(
        '--results-json',
        type=str,
        default=None,
        help='JSON path for detections when --source is a folder (default: <output>/results.json, or <source>_results.json)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=8,
        help='Images per model call when --source is a folder'
    )
    parser.add_argument(
        '--conf',
        type=float,
        default=0.5,
        help='Confidence threshold'
    )
    parser.add_argument(
        '--iou',
        type=float,
        default=0.45,
        help='IOU threshold for NMS'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cpu',
        choices=['cuda', 'cpu'],
        help='Device to use'
    )
    parser.add_argument(
        '--duration',
        type=int,
        default=30,
        help='Duration for webcam mode in seconds'
    )
    parser.add_argument(
        '--fps',
        type=int,
        default=30,
        help='FPS for video output'
    )
    parser.add_argument(
        '--class-id-map',
        type=str,
        default=None,
        help='Path to YAML file with class ID mapping (e.g., configs/convert_class_id_map.yaml)'
    )
    parser.add_argument(
        '--images-cache-dir',
        type=str,
        default=DEFAULT_IMAGES_CACHE_DIR,
        help='For --mode gcs_images: shared directory downloaded images are cached in - an '
             'image already there is reused instead of re-downloaded '
             f'(default: {DEFAULT_IMAGES_CACHE_DIR})'
    )

    args = parser.parse_args()
    main(args)
