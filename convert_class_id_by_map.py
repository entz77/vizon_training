"""CLI utility to convert class IDs in label files using a mapping from YAML config.

Usage examples:
    python convert_class_id_by_map.py --input-dir datasets/homecare/labels/train --map-config configs/convert_class_id_map.yaml
    python convert_class_id_by_map.py --input-dir in_labels --output-dir out_labels --map-config configs/convert_class_id_map.yaml
"""

import argparse
from pathlib import Path

from src.data.label_converter import convert_folder_class_id_by_map, load_class_id_map_from_yaml


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert class IDs in label files using a mapping from YAML configuration"
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
        "--map-config",
        type=str,
        required=True,
        help="Path to YAML file containing class ID mapping (e.g., configs/convert_class_id_map.yaml)",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="*.txt",
        help="Glob pattern for label files",
    )

    args = parser.parse_args()

    # Load the class ID mapping from YAML
    try:
        class_id_map = load_class_id_map_from_yaml(Path(args.map_config))
    except (FileNotFoundError, ValueError) as e:
        print(f"Error loading class ID map: {e}")
        return

    print(f"Loaded class ID mapping from {args.map_config}:")
    for orig_id, new_id in sorted(class_id_map.items()):
        print(f"  {orig_id} -> {new_id}")
    print()

    converted = convert_folder_class_id_by_map(
        input_dir=Path(args.input_dir),
        class_id_map=class_id_map,
        output_dir=Path(args.output_dir) if args.output_dir else None,
        pattern=args.pattern,
    )

    print(f"Converted {len(converted)} files.")
    for src, dst in converted:
        print(f"{src} -> {dst}")


if __name__ == "__main__":
    main()
