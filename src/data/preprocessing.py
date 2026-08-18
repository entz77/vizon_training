"""
This module contains functions for preprocessing the dataset, including:
- Converting annotations to the required format (e.g. YOLO TXT)
- Splitting a GCS-hosted dataset into training/validation/test sets
"""

import argparse
import random
import sys
from pathlib import Path

# When run directly as a script, the repo root is not on sys.path — fix that.
if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from google.cloud import storage

from src.data.label_converter import convert_line_xywhr_to_xywh

DEFAULT_IMAGES_BASE_PATH = "gs://prd-vveesee-edge-submit/images"


def _parse_gcs_uri(uri: str) -> tuple[str, str]:
    """Split a `gs://bucket/prefix` URI into (bucket_name, prefix)."""
    if not uri.startswith("gs://"):
        raise ValueError(f"Expected a gs:// URI, got '{uri}'")
    bucket_name, _, prefix = uri[len("gs://") :].partition("/")
    if not bucket_name:
        raise ValueError(f"Missing bucket name in URI: '{uri}'")
    return bucket_name, prefix.strip("/")


def _join_gcs_path(*parts: str) -> str:
    return "/".join(p.strip("/") for p in parts if p)


def _normalize_label_line(line: str) -> str:
    """Ensure one label line is in `class_id x y w h` format.

    Source labels may already be plain XYWH (5 values) or still carry a
    rotation field, XYWHR (6 values); only the latter needs conversion.
    """
    stripped = line.strip()
    if not stripped:
        return ""
    if len(stripped.split()) == 6:
        return convert_line_xywhr_to_xywh(stripped)
    return stripped


def split_dataset(
    source_dir: str,
    output_dir: str,
    train_pct: float = 0.8,
    val_pct: float = 0.2,
    test_pct: float = 0.0,
    image_extensions: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".webp"),
    images_base_path: str = DEFAULT_IMAGES_BASE_PATH,
    seed: int = 42,
) -> dict[str, int]:
    """
    Randomly split a GCS folder of `.txt` label files into train/val(/test) subsets,
    fetching each label's matching image from `images_base_path` by file stem.

    Label files are converted from `class_id x y w h r` to `class_id x y w h`
    (dropping the rotation field) before being written to the output location.
    Images are copied as-is.

    Output layout under `output_dir`:
        images/train/  images/val/  images/test/
        labels/train/  labels/val/  labels/test/
    (test subdirs are only created when test_pct > 0)

    Args:
        source_dir: `gs://bucket/prefix` folder containing the `.txt` label files.
        output_dir: `gs://bucket/prefix` root where split subsets will be written.
        train_pct: Fraction of samples for training (0-1).
        val_pct:   Fraction of samples for validation (0-1).
        test_pct:  Fraction of samples for test (0-1). Pass 0.0 to skip the test split.
        image_extensions: Candidate image extensions, tried in order, for each label.
        images_base_path: `gs://bucket/prefix` folder where source images live.
        seed: Random seed for reproducibility.

    Returns:
        Dict with keys "train", "val", and optionally "test" and their sample counts.
    """
    if abs(train_pct + val_pct + test_pct - 1.0) > 1e-6:
        raise ValueError(
            f"train_pct + val_pct + test_pct must equal 1.0, "
            f"got {train_pct + val_pct + test_pct:.4f}"
        )

    client = storage.Client()

    source_bucket_name, source_prefix = _parse_gcs_uri(source_dir)
    images_bucket_name, images_prefix = _parse_gcs_uri(images_base_path)
    images_bucket = client.bucket(images_bucket_name)

    print(f"Listing label files under '{source_dir}'...")
    label_blobs = [
        blob
        for blob in client.list_blobs(source_bucket_name, prefix=f"{source_prefix}/" if source_prefix else None)
        if blob.name.endswith(".txt")
    ]
    print(f"Found {len(label_blobs)} label files. Matching each to an image under '{images_base_path}'...")

    pairs: list[tuple[storage.Blob, storage.Blob, str]] = []
    for i, label_blob in enumerate(label_blobs, start=1):
        stem = Path(label_blob.name).stem
        # Label stems are prefixed with a task-unique hash, e.g. "0336bbc5__<image_stem>";
        # the image itself is stored under just "<image_stem>".
        stem = stem.partition("__")[2] or stem
        image_blob = None
        for ext in image_extensions:
            candidate = images_bucket.blob(_join_gcs_path(images_prefix, stem + ext))
            if candidate.exists():
                image_blob = candidate
                break
        if image_blob is not None:
            pairs.append((label_blob, image_blob, stem))
        if i % 25 == 0 or i == len(label_blobs):
            print(f"  matched {len(pairs)}/{i} (of {len(label_blobs)} labels)")

    if not pairs:
        raise FileNotFoundError(
            f"No label+image pairs found. Labels: '{source_dir}', images: '{images_base_path}'."
        )

    random.seed(seed)
    random.shuffle(pairs)

    n = len(pairs)
    n_train = round(n * train_pct)
    n_val = round(n * val_pct)

    splits: dict[str, list[tuple[storage.Blob, storage.Blob, str]]] = {
        "train": pairs[:n_train],
        "val": pairs[n_train : n_train + n_val],
    }
    if test_pct > 0:
        splits["test"] = pairs[n_train + n_val :]

    output_bucket_name, output_prefix = _parse_gcs_uri(output_dir)
    output_bucket = client.bucket(output_bucket_name)

    counts: dict[str, int] = {}
    for split_name, split_pairs in splits.items():
        print(f"Copying {len(split_pairs)} samples to '{output_dir}' ({split_name})...")
        for i, (label_blob, image_blob, stem) in enumerate(split_pairs, start=1):
            label_lines = label_blob.download_as_text().splitlines()
            converted = [_normalize_label_line(line) for line in label_lines]
            label_dest = output_bucket.blob(
                _join_gcs_path(output_prefix, "labels", split_name, stem + ".txt")
            )
            label_dest.upload_from_string("\n".join(converted) + "\n", content_type="text/plain")

            image_ext = Path(image_blob.name).suffix
            image_dest_name = _join_gcs_path(output_prefix, "images", split_name, stem + image_ext)
            images_bucket.copy_blob(image_blob, output_bucket, new_name=image_dest_name)

            if i % 25 == 0 or i == len(split_pairs):
                print(f"  [{split_name}] copied {i}/{len(split_pairs)}")

        counts[split_name] = len(split_pairs)

    summary = f"train: {counts['train']}, val: {counts['val']}"
    if "test" in counts:
        summary += f", test: {counts['test']}"
    print(f"Split {n} samples → {summary}")
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Split a GCS folder of label files into train/val(/test) subsets, fetching "
                    "each label's matching image from a GCS images folder. Labels are converted "
                    "from class_id x y w h r → class_id x y w h on the fly."
    )
    parser.add_argument("--source-dir", type=str, required=True,
                        help="gs://bucket/prefix folder containing .txt label files.")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="gs://bucket/prefix root where train/val/test subfolders will be created.")
    parser.add_argument("--images-base-path", type=str, default=DEFAULT_IMAGES_BASE_PATH,
                        help=f"gs://bucket/prefix folder containing source images "
                             f"(default: {DEFAULT_IMAGES_BASE_PATH}).")
    parser.add_argument("--train-pct", type=float, default=0.8,
                        help="Fraction of samples for training (default: 0.8).")
    parser.add_argument("--val-pct", type=float, default=0.2,
                        help="Fraction of samples for validation (default: 0.2).")
    parser.add_argument("--test-pct", type=float, default=0.0,
                        help="Fraction of samples for test (default: 0.0, skips test split).")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility (default: 42).")

    args = parser.parse_args()

    split_dataset(
        source_dir=args.source_dir,
        output_dir=args.output_dir,
        train_pct=args.train_pct,
        val_pct=args.val_pct,
        test_pct=args.test_pct,
        images_base_path=args.images_base_path,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
