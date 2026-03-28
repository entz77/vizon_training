"""CLI utility to convert label format from `class_id x y w h r` to YOLO `class_id x y w h`.

Usage examples:
    python convert_xywhr_to_xywh.py --input-dir datasets/homecare/labels/train
    python convert_xywhr_to_xywh.py --input-dir in_labels --output-dir out_labels
"""

import argparse
from pathlib import Path

from src.data.label_converter import convert_folder_xywhr_to_xywh


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert label files: class_id x y w h r -> class_id x y w h"
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Directory containing source label .txt files",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory for converted labels. If omitted, converts in-place.",
    )
    parser.add_argument(
        "--precision",
        type=int,
        default=6,
        help="Decimal precision for output coordinates",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="*.txt",
        help="Glob pattern for label files",
    )

    args = parser.parse_args()

    converted = convert_folder_xywhr_to_xywh(
        input_dir=Path(args.input_dir),
        output_dir=Path(args.output_dir) if args.output_dir else None,
        precision=args.precision,
        pattern=args.pattern,
    )

    print(f"Converted {len(converted)} files.")
    for src, dst in converted:
        print(f"{src} -> {dst}")


if __name__ == "__main__":
    main()