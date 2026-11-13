"""
Lensabonus detection inference pipeline.

Mirrors vizon_classifier_embedding/lensabonus_inference_pipeline.py, but runs this repo's
object detector (YOLO / RF-DETR, see src/models) instead of a crop classifier:

    1. Query Cloud SQL (transactions.user_activity) for capture_annotate rows in a time window.
    2. Download each row's image into the shared image folder (skipped when already cached - the
       same folder vizon_classifier_embedding uses, same on-disk naming).
    3. Run the detector locally on the downloaded images, in batches.
    4. Write results locally (results.json + YOLO label files + categories.json), the lensabonus-format
       annotation files for vizon_classifier_embedding and, only with --write-class-map, the model's
       class map to Cloud SQL labeling.class_maps. No labeling.labels rows are written here.

The lensabonus-format annotation files (--annotations-dir) are the hand-off to vizon_classifier_embedding:
run its lensabonus_inference_pipeline.py with --annotations-cache-dir pointing at that folder and it crops
and classifies OUR detections instead of the lensabonus ones (--classes then refers to this model's ids).

Usage:
    python lensabonus_inference_pipeline.py --weights runs/train/weights/best.pt --model-type yolo
    python lensabonus_inference_pipeline.py --weights best.pth --model-type rfdetr --model-size base \
        --start-ts 2026-06-17T00:00:00 --end-ts 2026-06-18T00:00:00 --conf 0.4
    python lensabonus_inference_pipeline.py --weights best.pt --model-name rtd_yolo_v3 --write-class-map --env prd
"""

import argparse
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Optional

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

from src.inference import DetectorPredictor
from src.models import available_models, build_model
from utils.cloud import download_blob, execute_query, list_blobs

DEFAULT_LOOKBACK_HOURS = 24
DEFAULT_OUTPUT_BASE_DIR = "infer/scheduled_runs"
# Shared with vizon_classifier_embedding (inference_common.DEFAULT_IMAGES_CACHE_DIR).
DEFAULT_IMAGES_CACHE_DIR = "/home/enitze/projects/repos/vizon/shared_image_folder"
DEFAULT_ANNOTATIONS_BASE_DIR = "infer/annotations"
IMAGES_BASE_PATH = {"dev": "gs://dev-vveesee-edge-submit", "prd": "gs://prd-vveesee-edge-submit"}
IMAGES_GCS_SUBFOLDER = "images"
BOX_DECIMALS = 4  # max decimal digits of the normalized box coordinates in the annotation files

ACTIVITY_LIST_QUERY = """
    SELECT id, data
    FROM transactions.user_activity
    WHERE activity = 'capture_annotate'
      AND status = 'uploading'
      AND created_at BETWEEN %s AND %s
    ORDER BY created_at
"""
CLASS_MAPS_TABLE = "labeling.class_maps"


def get_file_entry(files: List[dict], file_id: str) -> Optional[dict]:
    return next((f for f in files if f.get("file_id") == file_id), None)


def fetch_activity_rows(start_ts: datetime, end_ts: datetime, env: str) -> List[dict]:
    """capture_annotate rows in [start_ts, end_ts] that have an "image" file entry."""
    records = json.loads(execute_query(ACTIVITY_LIST_QUERY, (start_ts, end_ts), env=env))
    rows = []
    for record in records:
        files = (record["data"] or {}).get("files")
        if isinstance(files, list) and get_file_entry(files, "image"):
            rows.append({"id": record["id"], "data": record["data"]})
    return rows


def find_cached_activity_file(cache_dir: Path, row_id: str, filename: str) -> Optional[Path]:
    """Already-downloaded "{row_id}_<upload_ts>_{filename}" in cache_dir - a plain glob, no GCS call."""
    return next((p for p in cache_dir.glob(f"{row_id}_*_{filename}") if p.is_file()), None)


def resolve_gcs_blob_path(images_base_path: str, row_id: str, filename: str) -> str:
    """Files are uploaded as images/{row_id}_{upload_ts}_{filename}; the timestamp isn't queryable, so list by prefix."""
    prefix = f"{images_base_path.rstrip('/')}/{IMAGES_GCS_SUBFOLDER}/{row_id}_"
    matches = [b for b in list_blobs(prefix) if b.endswith(f"_{filename}")]
    if not matches:
        raise FileNotFoundError(f"No GCS object found matching {prefix}*_{filename}")
    return matches[0]


def download_images(rows: List[dict], images_base_path: str, cache_dir: Path) -> List[dict]:
    """
    Ensure every row's image is in cache_dir, downloading only the missing ones. All downloads
    finish before inference starts. Rows whose image can't be found are skipped with a warning.

    Returns [{"id", "image_path"}]; image_path.stem is the labels-table image_id (same
    convention as the embedding pipeline, so rows from both pipelines join on it).
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    entries, downloaded = [], 0
    for row in tqdm(rows, desc="Fetching images", unit="img"):
        filename = get_file_entry(row["data"]["files"], "image")["filename"]
        path = find_cached_activity_file(cache_dir, row["id"], filename)
        if path is None:
            try:
                blob = resolve_gcs_blob_path(images_base_path, row["id"], filename)
                path = download_blob(blob, cache_dir / blob.rsplit("/", 1)[-1])
                downloaded += 1
            except FileNotFoundError as e:
                print(f"  Warning: {e}, skipping", file=sys.stderr)
                continue
        entries.append({"id": row["id"], "image_path": path})
    print(f"  {downloaded} downloaded, {len(entries) - downloaded} already cached")
    return entries


def run_detection(predictor: DetectorPredictor, entries: List[dict], batch_size: int) -> List[dict]:
    """Detect on every entry's image, batch_size images per model call. Returns one record per entry."""
    records = []
    for i in tqdm(range(0, len(entries), batch_size), desc="Detecting", unit="batch"):
        chunk = entries[i:i + batch_size]
        results = predictor.predict_batch([str(e["image_path"]) for e in chunk], verbose=False)
        for entry, result in zip(chunk, results):
            records.append({"entry": entry, **result})
    return records


def build_class_map(model) -> dict:
    """
    {class_id (str): class_name} in the ids the annotation/label files use: the model's native ids
    after its class_id_map (if any), so the class map always matches the written class_ids.
    """
    names = model.get_class_names()
    id_map = getattr(model, "class_id_map", None) or {}
    return {str(id_map.get(cid, cid)): name for cid, name in sorted(names.items())}


def write_local_outputs(records: List[dict], class_map: dict, run_dir: Path, model_tag: str) -> None:
    """results.json, YOLO-format labels/<image_id>.txt (class x_center y_center w h) and categories.json."""
    labels_dir = run_dir / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)

    out = []
    for rec in records:
        entry = rec["entry"]
        image_id = entry["image_path"].stem
        lines = []
        for d in rec["detections"]:
            b = d["bbox_normalized"]
            lines.append(f"{d['class_id']} {b['x_center']:.6f} {b['y_center']:.6f} {b['width']:.6f} {b['height']:.6f}")
        (labels_dir / f"{image_id}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
        out.append({
            "row_id": entry["id"], "image_id": image_id, "image_path": str(entry["image_path"]),
            "image_shape": rec["image_shape"], "detections": rec["detections"],
        })

    (run_dir / "results.json").write_text(json.dumps(out, indent=2))
    (run_dir / "categories.json").write_text(json.dumps(
        {"categories": [{"id": int(k), "name": v} for k, v in sorted(class_map.items(), key=lambda kv: int(kv[0]))]},
        indent=2,
    ))


def write_lensabonus_annotations(records: List[dict], annotations_dir: Path, model_name: str) -> int:
    """
    Write one raw-lensabonus-format annotation file per image, so vizon_classifier_embedding's
    lensabonus_inference_pipeline.py can crop from OUR detections instead of lensabonus's own:

        {"model_name": ..., "model_predictions": [{"class_id": 0, "box": {left, top, right, bottom}}]}

    Boxes are normalized (0-1), rounded to BOX_DECIMALS (4) decimal digits. Files are named "<image stem>.json" = "{row_id}_{upload_ts}_{stem}.json",
    which is exactly what that pipeline's find_cached_activity_file globs for ("{row_id}_*_{filename}",
    filename being the row's annotations filename, which shares the image's stem) - so pointing its
    --annotations-cache-dir here makes it skip the GCS annotation download and use these.
    """
    annotations_dir.mkdir(parents=True, exist_ok=True)
    for rec in records:
        h, w = rec["image_shape"]
        preds = [
            {
                "class_id": d["class_id"],
                "box": {
                    "left": round(max(0.0, d["bbox"]["x1"] / w), BOX_DECIMALS),
                    "top": round(max(0.0, d["bbox"]["y1"] / h), BOX_DECIMALS),
                    "right": round(min(1.0, d["bbox"]["x2"] / w), BOX_DECIMALS),
                    "bottom": round(min(1.0, d["bbox"]["y2"] / h), BOX_DECIMALS),
                },
            }
            for d in rec["detections"]
        ]
        path = annotations_dir / f"{rec['entry']['image_path'].stem}.json"
        path.write_text(json.dumps({"model_name": model_name, "model_predictions": preds}))
    return len(records)


def write_class_map_to_sql(class_map: dict, map_id: str, env: str) -> None:
    """
    Upsert this model's class map ({"0": "name", ...}) as a labeling.class_maps row keyed by map_id
    (= --model-name). vizon_classifier_embedding writes the raw-detection labels rows with
    class_map = the annotation files' model_name but never writes a class map for it - it only
    reads it (to fill box_labels.object_class_map_class) - so this is where it comes from.
    Idempotent: map_id is the primary key. created_at/updated_at are managed by Postgres.
    """
    execute_query(
        f"INSERT INTO {CLASS_MAPS_TABLE} (map_id, class_map) VALUES (%s, %s) "
        f"ON CONFLICT (map_id) DO UPDATE SET class_map = EXCLUDED.class_map",
        (map_id, json.dumps(class_map)), env=env,
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Pull capture_annotate images from Cloud SQL + GCS and run the object detector locally.")
    p.add_argument("--weights", required=True, help="Path to trained detector weights")
    p.add_argument("--model-type", choices=available_models(), default="yolo", help="Detector backend (default: yolo)")
    p.add_argument("--model-size", default=None, help="Model size (backend default if omitted)")
    p.add_argument("--device", default=None, help="cpu / cuda (auto if omitted)")
    p.add_argument("--class-id-map", default=None, help="Optional YAML class-id mapping applied to predictions")
    p.add_argument("--conf", type=float, default=0.5, help="Confidence threshold (default: 0.5)")
    p.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold, YOLO only (default: 0.45)")
    p.add_argument("--batch-size", type=int, default=8, help="Images per model call (default: 8)")
    p.add_argument("--start-ts", default=None, help=f"ISO datetime lower bound (default: {DEFAULT_LOOKBACK_HOURS}h before --end-ts)")
    p.add_argument("--end-ts", default=None, help="ISO datetime upper bound (default: now)")
    p.add_argument("--env", choices=["dev", "prd"], default="dev", help="Environment: Cloud SQL instance + GCS bucket (default: dev)")
    p.add_argument("--images-cache-dir", default=DEFAULT_IMAGES_CACHE_DIR,
                   help=f"Shared image folder (default: {DEFAULT_IMAGES_CACHE_DIR})")
    p.add_argument("--output-base-dir", default=DEFAULT_OUTPUT_BASE_DIR,
                   help=f"Base folder for this run's timestamped output folder (default: {DEFAULT_OUTPUT_BASE_DIR})")
    p.add_argument("--model-name", default=None,
                   help="Stable identifier for this trained model, e.g. rtd_yolo_v3. Used as the annotation files' "
                        "model_name (-> agent/class_map in labeling.labels via the embedding pipeline) and as the "
                        "default annotations folder name. Default: <model-type>_<weights file name>, which is NOT unique "
                        "across trainings (every YOLO run saves best.pt) - set this explicitly for real runs.")
    p.add_argument("--annotations-dir", default=None,
                   help="Where lensabonus-format annotation files are written for vizon_classifier_embedding "
                        f"(default: {DEFAULT_ANNOTATIONS_BASE_DIR}/<model tag>)")
    p.add_argument("--write-class-map", action="store_true",
                   help="Upsert this model's class map into Cloud SQL labeling.class_maps (map_id = model name). "
                        "Nothing else is written to the database - labeling.labels rows come from the "
                        "vizon_classifier_embedding pipeline. Off by default.")
    args = p.parse_args()
    return args


def main():
    args = parse_args()

    end_ts = datetime.fromisoformat(args.end_ts) if args.end_ts else datetime.now(timezone.utc)
    start_ts = datetime.fromisoformat(args.start_ts) if args.start_ts else end_ts - timedelta(hours=DEFAULT_LOOKBACK_HOURS)

    run_dir = Path(args.output_base_dir) / datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    print(f"Run folder: {run_dir}")

    print(f"\nStep 1: Querying Cloud SQL ({args.env}) for capture_annotate rows ({start_ts} to {end_ts})...")
    rows = fetch_activity_rows(start_ts, end_ts, args.env)
    print(f"  Found {len(rows)} row(s)")
    if not rows:
        print("No rows found in the given time range.")
        return

    print(f"\nStep 2: Fetching images (shared folder: {args.images_cache_dir})...")
    entries = download_images(rows, IMAGES_BASE_PATH[args.env], Path(args.images_cache_dir))
    if not entries:
        print("No images were available.")
        return

    print(f"\nStep 3: Loading {args.model_type} detector and running local inference...")
    model_kwargs = {"device": args.device} if args.device else {}
    model = build_model(args.model_type, args.model_size, **model_kwargs)
    model.load_weights(args.weights)
    if args.class_id_map:
        model.load_class_id_map(args.class_id_map)
    predictor = DetectorPredictor(model=model, conf_threshold=args.conf, iou_threshold=args.iou)
    records = run_detection(predictor, entries, args.batch_size)
    total = sum(len(r["detections"]) for r in records)
    print(f"  {total} detection(s) in {len(records)} image(s)")

    print("\nStep 4: Writing local output...")
    model_tag = args.model_name or f"{args.model_type}_{Path(args.weights).stem}"
    if not args.model_name:
        print(f"  Warning: no --model-name given, using '{model_tag}' - not unique across trainings", file=sys.stderr)
    class_map = build_class_map(model)
    write_local_outputs(records, class_map, run_dir, model_tag)
    if not class_map:
        print("  Warning: model reported no class names, class map is empty", file=sys.stderr)
    print(f"  Wrote results.json, categories.json and {len(records)} label file(s) to {run_dir}")

    annotations_dir = Path(args.annotations_dir or Path(DEFAULT_ANNOTATIONS_BASE_DIR) / model_tag)
    write_lensabonus_annotations(records, annotations_dir, model_tag)
    print(f"  Wrote {len(records)} lensabonus-format annotation file(s) to {annotations_dir}")

    if args.write_class_map:
        print(f"\nStep 5: Writing class map '{model_tag}' to {CLASS_MAPS_TABLE} ({args.env})...")
        write_class_map_to_sql(class_map, model_tag, args.env)
        print(f"  Upserted {len(class_map)} class(es)")

    print(f"\nDone. Local run folder: {run_dir}")


if __name__ == "__main__":
    main()
