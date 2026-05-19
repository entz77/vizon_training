"""
This module contains functions for preprocessing the dataset, including:
- Converting annotations to the required format (e.g. YOLO TXT)
- Splitting the dataset into training/validation/test sets
"""

import argparse
import random
import shutil
import sys
from pathlib import Path

# When run directly as a script, the repo root is not on sys.path — fix that.
if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.label_converter import convert_file_xywhr_to_xywh


def split_dataset(
    source_dir: str | Path,
    output_dir: str | Path,
    train_pct: float = 0.8,
    val_pct: float = 0.2,
    test_pct: float = 0.0,
    image_extensions: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".webp"),
    seed: int = 42,
) -> dict[str, int]:
    """
    Randomly split a folder of image+TXT annotation pairs into train/val(/test) subsets.

    Label files are converted from `class_id x y w h r` to `class_id x y w h`
    (dropping the rotation field) before being written to the target directory.
    Images are copied as-is.

    The function expects each image file to have a matching .txt annotation with the
    same stem in the same directory. Output layout:
        output_dir/
            images/train/  images/val/  images/test/
            labels/train/  labels/val/  labels/test/
        (test subdirs are only created when test_pct > 0)

    Args:
        source_dir: Folder that contains both images and .txt label files.
        output_dir: Root folder where split subsets will be written.
        train_pct: Fraction of samples for training (0–1).
        val_pct:   Fraction of samples for validation (0–1).
        test_pct:  Fraction of samples for test (0–1). Pass 0.0 to skip the test split.
        image_extensions: File extensions considered as images.
        seed: Random seed for reproducibility.

    Returns:
        Dict with keys "train", "val", and optionally "test" and their sample counts.
    """
    if abs(train_pct + val_pct + test_pct - 1.0) > 1e-6:
        raise ValueError(
            f"train_pct + val_pct + test_pct must equal 1.0, "
            f"got {train_pct + val_pct + test_pct:.4f}"
        )

    source_dir = Path(source_dir)
    output_dir = Path(output_dir)

    pairs: list[tuple[Path, Path]] = []
    for img_path in sorted(source_dir.iterdir()):
        if img_path.suffix.lower() not in image_extensions:
            continue
        label_path = img_path.with_suffix(".txt")
        if not label_path.exists():
            continue
        pairs.append((img_path, label_path))

    if not pairs:
        raise FileNotFoundError(
            f"No image+label pairs found in '{source_dir}'. "
            f"Ensure each image has a matching .txt file."
        )

    random.seed(seed)
    random.shuffle(pairs)

    n = len(pairs)
    n_train = round(n * train_pct)
    n_val = round(n * val_pct)

    splits: dict[str, list[tuple[Path, Path]]] = {
        "train": pairs[:n_train],
        "val": pairs[n_train : n_train + n_val],
    }
    if test_pct > 0:
        splits["test"] = pairs[n_train + n_val :]

    counts: dict[str, int] = {}
    for split_name, split_pairs in splits.items():
        img_out = output_dir / "images" / split_name
        lbl_out = output_dir / "labels" / split_name
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)

        for img_path, label_path in split_pairs:
            shutil.copy2(img_path, img_out / img_path.name)
            convert_file_xywhr_to_xywh(
                input_path=label_path,
                output_path=lbl_out / label_path.name,
            )

        counts[split_name] = len(split_pairs)

    summary = f"train: {counts['train']}, val: {counts['val']}"
    if "test" in counts:
        summary += f", test: {counts['test']}"
    print(f"Split {n} samples → {summary}")
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Split a flat folder of image+label pairs into train/val(/test) subsets. "
                    "Labels are converted from class_id x y w h r → class_id x y w h on the fly."
    )
    parser.add_argument("--source-dir", type=str, required=True,
                        help="Folder containing images and matching .txt label files.")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Root folder where train/val/test subfolders will be created.")
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
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
