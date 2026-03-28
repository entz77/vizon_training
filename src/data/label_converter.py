"""Utilities for converting rotated bbox labels to YOLO OBB corner format."""

import math
from pathlib import Path
from typing import List, Tuple


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