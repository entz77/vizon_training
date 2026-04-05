"""Utilities for converting rotated bbox labels to YOLO OBB corner format."""

import math
from pathlib import Path
from typing import Dict, List, Tuple

import yaml


def _rotated_box_to_corners(x: float, y: float, w: float, h: float, r: float) -> List[Tuple[float, float]]:
    """Convert center-width-height-angle to four corners.

    Args:
        x: center x
        y: center y
        w: width
        h: height
        r: rotation angle in radians

    Returns:
        List of 4 points in clockwise order: [(x1, y1), ..., (x4, y4)]
    """
    hw = w / 2.0
    hh = h / 2.0

    # Unrotated rectangle corners around origin in clockwise order.
    local_corners = [(-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh)]

    cos_r = math.cos(r)
    sin_r = math.sin(r)

    corners = []
    for dx, dy in local_corners:
        cx = x + dx * cos_r - dy * sin_r
        cy = y + dx * sin_r + dy * cos_r
        corners.append((cx, cy))

    return corners


def convert_line_xywhr_to_obb(line: str, angle_unit: str = "radians", precision: int = 6) -> str:
    """Convert one label line from `class_id x y w h r` to YOLO OBB corners.

    Input line format:
        class_id x y w h r

    Output line format:
        class_id x1 y1 x2 y2 x3 y3 x4 y4

    Args:
        line: Label line text.
        angle_unit: "radians" or "degrees".
        precision: Decimal precision for output coordinates.

    Returns:
        Converted line (or original blank line).
    """
    stripped = line.strip()
    if not stripped:
        return ""

    parts = stripped.split()
    if len(parts) != 6:
        raise ValueError(f"Expected 6 values per line, got {len(parts)}: '{line.rstrip()}'")

    class_id = parts[0]
    x, y, w, h, r = map(float, parts[1:])

    if angle_unit not in {"radians", "degrees"}:
        raise ValueError("angle_unit must be 'radians' or 'degrees'")

    if angle_unit == "degrees":
        r = math.radians(r)

    corners = _rotated_box_to_corners(x=x, y=y, w=w, h=h, r=r)

    coord_values: List[str] = []
    for px, py in corners:
        coord_values.append(f"{px:.{precision}f}")
        coord_values.append(f"{py:.{precision}f}")

    return " ".join([class_id, *coord_values])


def convert_line_xywhr_to_xywh(line: str, precision: int = 6) -> str:
    """Convert one label line from `class_id x y w h r` to standard YOLO XYWH format.

    Input line format:
        class_id x y w h r

    Output line format:
        class_id x y w h
    """
    stripped = line.strip()
    if not stripped:
        return ""

    parts = stripped.split()
    if len(parts) != 6:
        raise ValueError(f"Expected 6 values per line, got {len(parts)}: '{line.rstrip()}'")

    class_id = parts[0]
    x, y, w, h, _ = map(float, parts[1:])

    return " ".join(
        [
            class_id,
            f"{x:.{precision}f}",
            f"{y:.{precision}f}",
            f"{w:.{precision}f}",
            f"{h:.{precision}f}",
        ]
    )


def convert_file_xywhr_to_obb(
    input_path: Path,
    output_path: Path = None,
    angle_unit: str = "radians",
    precision: int = 6,
) -> Path:
    """Convert all lines in one label file from XYWHR format to YOLO OBB format."""
    input_path = Path(input_path)
    output_path = Path(output_path) if output_path else input_path

    lines = input_path.read_text(encoding="utf-8").splitlines()
    converted = [convert_line_xywhr_to_obb(line, angle_unit=angle_unit, precision=precision) for line in lines]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(converted) + "\n", encoding="utf-8")
    return output_path


def convert_file_xywhr_to_xywh(
    input_path: Path,
    output_path: Path = None,
    precision: int = 6,
) -> Path:
    """Convert all lines in one label file from XYWHR format to standard YOLO XYWH format."""
    input_path = Path(input_path)
    output_path = Path(output_path) if output_path else input_path

    lines = input_path.read_text(encoding="utf-8").splitlines()
    converted = [convert_line_xywhr_to_xywh(line, precision=precision) for line in lines]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(converted) + "\n", encoding="utf-8")
    return output_path


def convert_folder_xywhr_to_obb(
    input_dir: Path,
    output_dir: Path = None,
    angle_unit: str = "radians",
    precision: int = 6,
    pattern: str = "*.txt",
) -> List[Tuple[Path, Path]]:
    """Convert every matching label file in a folder.

    If output_dir is None, conversion happens in-place.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir) if output_dir else input_dir

    if not input_dir.exists() or not input_dir.is_dir():
        raise ValueError(f"Invalid input directory: {input_dir}")

    results: List[Tuple[Path, Path]] = []
    for src in sorted(input_dir.glob(pattern)):
        if not src.is_file():
            continue

        dst = output_dir / src.name
        converted_path = convert_file_xywhr_to_obb(
            input_path=src,
            output_path=dst,
            angle_unit=angle_unit,
            precision=precision,
        )
        results.append((src, converted_path))

    return results


def convert_folder_xywhr_to_xywh(
    input_dir: Path,
    output_dir: Path = None,
    precision: int = 6,
    pattern: str = "*.txt",
) -> List[Tuple[Path, Path]]:
    """Convert every matching label file in a folder to standard YOLO XYWH format.

    If output_dir is None, conversion happens in-place.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir) if output_dir else input_dir

    if not input_dir.exists() or not input_dir.is_dir():
        raise ValueError(f"Invalid input directory: {input_dir}")

    results: List[Tuple[Path, Path]] = []
    for src in sorted(input_dir.glob(pattern)):
        if not src.is_file():
            continue

        dst = output_dir / src.name
        converted_path = convert_file_xywhr_to_xywh(
            input_path=src,
            output_path=dst,
            precision=precision,
        )
        results.append((src, converted_path))

    return results


def convert_line_class_id_to_zero(line: str) -> str:
    """Convert class ID in a label line to 0, keeping all coordinates unchanged.

    Input line format:
        class_id x1 y1 ... (any format with at least the first value as class_id)

    Output line format:
        0 x1 y1 ... (all coordinates preserved)

    Args:
        line: Label line text.

    Returns:
        Converted line with class_id replaced by 0 (or original blank line).
    """
    stripped = line.strip()
    if not stripped:
        return ""

    parts = stripped.split()
    if len(parts) < 2:
        raise ValueError(f"Expected at least 2 values per line, got {len(parts)}: '{line.rstrip()}'")

    # Replace class_id with 0, keep all other values unchanged
    return " ".join(["0", *parts[1:]])


def convert_file_class_id_to_zero(
    input_path: Path,
    output_path: Path = None,
) -> Path:
    """Convert all class IDs in one label file to 0."""
    input_path = Path(input_path)
    output_path = Path(output_path) if output_path else input_path

    lines = input_path.read_text(encoding="utf-8").splitlines()
    converted = [convert_line_class_id_to_zero(line) for line in lines]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(converted) + "\n", encoding="utf-8")
    return output_path


def convert_folder_class_id_to_zero(
    input_dir: Path,
    output_dir: Path = None,
    pattern: str = "*.txt",
) -> List[Tuple[Path, Path]]:
    """Convert all class IDs to 0 in every matching label file in a folder.

    If output_dir is None, conversion happens in-place.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir) if output_dir else input_dir

    if not input_dir.exists() or not input_dir.is_dir():
        raise ValueError(f"Invalid input directory: {input_dir}")

    results: List[Tuple[Path, Path]] = []
    for src in sorted(input_dir.glob(pattern)):
        if not src.is_file():
            continue

        dst = output_dir / src.name
        converted_path = convert_file_class_id_to_zero(
            input_path=src,
            output_path=dst,
        )
        results.append((src, converted_path))

    return results


def convert_line_class_id_by_map(line: str, class_id_map: Dict[int, int]) -> str:
    """Convert class ID in a label line using a mapping dictionary.

    Input line format:
        class_id x1 y1 ... (any format with at least the first value as class_id)

    Output line format:
        mapped_class_id x1 y1 ... (all coordinates preserved)

    Args:
        line: Label line text.
        class_id_map: Dictionary mapping original class IDs to new class IDs.

    Returns:
        Converted line with class_id replaced by mapped value (or original blank line).
        If class_id is not in the map, the original class_id is kept.

    Raises:
        ValueError: If line doesn't have at least 2 values.
    """
    stripped = line.strip()
    if not stripped:
        return ""

    parts = stripped.split()
    if len(parts) < 2:
        raise ValueError(f"Expected at least 2 values per line, got {len(parts)}: '{line.rstrip()}'")

    original_class_id = int(parts[0])
    new_class_id = class_id_map.get(original_class_id, original_class_id)

    return " ".join([str(new_class_id), *parts[1:]])


def convert_file_class_id_by_map(
    input_path: Path,
    class_id_map: Dict[int, int],
    output_path: Path = None,
) -> Path:
    """Convert all class IDs in one label file using a mapping dictionary."""
    input_path = Path(input_path)
    output_path = Path(output_path) if output_path else input_path

    lines = input_path.read_text(encoding="utf-8").splitlines()
    converted = [convert_line_class_id_by_map(line, class_id_map) for line in lines]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(converted) + "\n", encoding="utf-8")
    return output_path


def convert_folder_class_id_by_map(
    input_dir: Path,
    class_id_map: Dict[int, int],
    output_dir: Path = None,
    pattern: str = "*.txt",
) -> List[Tuple[Path, Path]]:
    """Convert all class IDs using a mapping in every matching label file in a folder.

    If output_dir is None, conversion happens in-place.

    Args:
        input_dir: Directory containing source label files.
        class_id_map: Dictionary mapping original class IDs to new class IDs.
        output_dir: Directory for converted labels. If None, converts in-place.
        pattern: Glob pattern for label files.

    Returns:
        List of tuples (source_path, destination_path) for all converted files.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir) if output_dir else input_dir

    if not input_dir.exists() or not input_dir.is_dir():
        raise ValueError(f"Invalid input directory: {input_dir}")

    results: List[Tuple[Path, Path]] = []
    for src in sorted(input_dir.glob(pattern)):
        if not src.is_file():
            continue

        dst = output_dir / src.name
        converted_path = convert_file_class_id_by_map(
            input_path=src,
            class_id_map=class_id_map,
            output_path=dst,
        )
        results.append((src, converted_path))

    return results


def load_class_id_map_from_yaml(yaml_path: Path) -> Dict[int, int]:
    """Load class ID mapping from a YAML configuration file.

    Expected YAML format:
        class_id_map:
          0: 2
          1: 1
          2: 0

    Args:
        yaml_path: Path to the YAML mapping file.

    Returns:
        Dictionary mapping original class IDs to new class IDs.

    Raises:
        FileNotFoundError: If YAML file doesn't exist.
        ValueError: If YAML format is invalid or 'class_id_map' key is missing.
    """
    yaml_path = Path(yaml_path)
    if not yaml_path.exists():
        raise FileNotFoundError(f"YAML mapping file not found: {yaml_path}")

    with open(yaml_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if not isinstance(config, dict) or "class_id_map" not in config:
        raise ValueError(f"Invalid YAML format. Expected 'class_id_map' key at root level in {yaml_path}")

    class_id_map_raw = config["class_id_map"]
    if not isinstance(class_id_map_raw, dict):
        raise ValueError(f"'class_id_map' must be a dictionary in {yaml_path}")

    # Convert all keys and values to integers
    return {int(k): int(v) for k, v in class_id_map_raw.items()}