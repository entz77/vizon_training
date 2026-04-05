"""Crop bounding boxes from images and save organized by class."""

import argparse
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np


class BBoxCropper:
    """Crop and save bounding boxes from images using YOLO format labels."""

    def __init__(
        self,
        images_dir: Path,
        labels_dir: Path,
        output_dir: Path,
        classes_file: Path,
        padding: int = 0,
        config_file: Path = None,
    ):
        """Initialize the BBox cropper.

        Args:
            images_dir: Directory containing images
            labels_dir: Directory containing YOLO format label files
            output_dir: Directory to save cropped images
            classes_file: Path to classes.txt file (one class per line)
            padding: Pixel padding around crops (default: 0)
            config_file: Optional path to bbox_crop_config.yaml for filtering classes
        """
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        self.output_dir = Path(output_dir)
        self.classes_file = Path(classes_file)
        self.padding = padding

        # Load all class names
        self.all_classes = self._load_classes()

        # Load included classes from config (if provided)
        self.include_classes = self._load_config(config_file)

        # Filter classes based on config
        self.classes = self._filter_classes()

        # Create output directories for each included class
        self._create_output_dirs()

        # Statistics
        self.stats = {
            "images_processed": 0,
            "crops_saved": 0,
            "crops_per_class": {cls: 0 for cls in self.classes},
        }

    def _load_config(self, config_file: Path) -> List[str]:
        """Load include_classes from config file.

        Args:
            config_file: Path to bbox_crop_config.yaml

        Returns:
            List of class names to include, or empty list if no config
        """
        if config_file is None:
            return []

        config_file = Path(config_file)
        if not config_file.exists():
            print(f"Warning: Config file not found: {config_file}")
            return []

        with open(config_file, "r") as f:
            config = yaml.safe_load(f)

        if config is None or "include_classes" not in config:
            return []

        include_classes = config.get("include_classes", [])
        return [cls.strip() for cls in include_classes if cls]

    def _filter_classes(self) -> List[str]:
        """Filter classes based on include_classes config.

        Returns:
            Filtered list of classes to process
        """
        if not self.include_classes:
            # If no filter specified, include all classes
            return self.all_classes

        # Only include classes that are in both include_classes and all_classes
        filtered = [cls for cls in self.all_classes if cls in self.include_classes]

        if len(filtered) < len(self.include_classes):
            missing = set(self.include_classes) - set(self.all_classes)
            if missing:
                print(f"Warning: Classes in config not found in classes.txt: {missing}")

        return filtered

    def _load_classes(self) -> List[str]:
        """Load class names from classes.txt file.

        Returns:
            List of class names
        """
        if not self.classes_file.exists():
            raise FileNotFoundError(f"Classes file not found: {self.classes_file}")

        with open(self.classes_file, "r") as f:
            classes = [line.strip() for line in f if line.strip()]

        return classes

    def _create_output_dirs(self) -> None:
        """Create output directories for each included class."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        for class_name in self.classes:
            class_dir = self.output_dir / class_name
            class_dir.mkdir(parents=True, exist_ok=True)

        # Log which classes are being processed
        if self.include_classes:
            print(f"Processing classes: {', '.join(self.classes)}")
        else:
            print(f"Processing all classes: {', '.join(self.classes)}")

    def _parse_yolo_line(self, line: str, img_width: int, img_height: int) -> Tuple[int, int, int, int, int]:
        """Parse a YOLO format label line and convert to pixel coordinates.

        Format: class_id x_center y_center width height (all normalized 0-1)

        Args:
            line: Label line string
            img_width: Image width in pixels
            img_height: Image height in pixels

        Returns:
            Tuple of (class_id, x1, y1, x2, y2) in pixel coordinates
        """
        parts = line.strip().split()
        if len(parts) < 5:
            return None

        try:
            class_id = int(parts[0])
            x_center = float(parts[1])
            y_center = float(parts[2])
            width = float(parts[3])
            height = float(parts[4])

            # Convert from normalized center coordinates to pixel coordinates
            x1 = int((x_center - width / 2) * img_width)
            y1 = int((y_center - height / 2) * img_height)
            x2 = int((x_center + width / 2) * img_width)
            y2 = int((y_center + height / 2) * img_height)

            return class_id, x1, y1, x2, y2
        except (ValueError, IndexError):
            return None

    def _crop_bbox(self, image: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> np.ndarray:
        """Crop a bounding box from an image with padding.

        Args:
            image: Image array (H x W x C)
            x1: Left x coordinate
            y1: Top y coordinate
            x2: Right x coordinate
            y2: Bottom y coordinate

        Returns:
            Cropped image array
        """
        h, w = image.shape[:2]

        # Apply padding with boundary checks
        x1_padded = max(0, x1 - self.padding)
        y1_padded = max(0, y1 - self.padding)
        x2_padded = min(w, x2 + self.padding)
        y2_padded = min(h, y2 + self.padding)

        crop = image[y1_padded:y2_padded, x1_padded:x2_padded]
        return crop

    def process_folder(self) -> dict:
        """Process all images and labels in the folder.

        Returns:
            Dictionary with processing statistics
        """
        image_files = sorted(self.images_dir.glob("*"))
        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
        image_files = [f for f in image_files if f.suffix.lower() in image_extensions]

        print(f"Processing {len(image_files)} images...")

        for image_file in image_files:
            # Find corresponding label file
            label_file = self.labels_dir / (image_file.stem + ".txt")

            if not label_file.exists():
                print(f"  Warning: Label file not found for {image_file.name}")
                continue

            self._process_image(image_file, label_file)

        return self.stats

    def _process_image(self, image_file: Path, label_file: Path) -> None:
        """Process a single image and its labels.

        Args:
            image_file: Path to image file
            label_file: Path to label file
        """
        # Read image
        image = cv2.imread(str(image_file))
        if image is None:
            print(f"  Error: Could not read image {image_file.name}")
            return

        h, w = image.shape[:2]

        # Read labels
        with open(label_file, "r") as f:
            lines = f.readlines()

        if not lines:
            return

        crop_count = 0
        for line in lines:
            bbox_info = self._parse_yolo_line(line, w, h)
            if bbox_info is None:
                continue

            class_id, x1, y1, x2, y2 = bbox_info

            # Validate class_id against all classes
            if class_id < 0 or class_id >= len(self.all_classes):
                print(f"  Warning: Invalid class_id {class_id} in {label_file.name}")
                continue

            # Get the class name
            class_name = self.all_classes[class_id]

            # Skip if this class is not in the filtered include list
            if class_name not in self.classes:
                continue

            # Crop the bounding box
            crop = self._crop_bbox(image, x1, y1, x2, y2)

            # Skip very small crops
            if crop.shape[0] < 2 or crop.shape[1] < 2:
                continue
            output_path = self.output_dir / class_name / f"{image_file.stem}_{crop_count}.jpg"

            success = cv2.imwrite(str(output_path), crop)
            if success:
                crop_count += 1
                self.stats["crops_saved"] += 1
                self.stats["crops_per_class"][class_name] += 1

        if crop_count > 0:
            self.stats["images_processed"] += 1
            print(f"  ✓ {image_file.name}: {crop_count} crops saved")

    def print_summary(self) -> None:
        """Print processing summary."""
        print("\n" + "=" * 60)
        print("PROCESSING SUMMARY")
        print("=" * 60)
        print(f"Images processed: {self.stats['images_processed']}")
        print(f"Total crops saved: {self.stats['crops_saved']}")
        print("\nCrops per class:")
        for class_name, count in self.stats["crops_per_class"].items():
            if count > 0:
                print(f"  {class_name}: {count}")
        print("=" * 60)


def main():
    """Command-line interface for bbox cropper."""
    parser = argparse.ArgumentParser(
        description="Crop bounding boxes from images and save organized by class",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Crop all classes from homecare dataset
  python bbox_cropper.py \\
    --images-dir datasets/homecare/images/train \\
    --labels-dir datasets/homecare/labels/train \\
    --output-dir ./crops_homecare \\
    --classes-file datasets/homecare/classes.txt

  # Crop with config file to filter specific classes
  python bbox_cropper.py \\
    --images-dir datasets/homecare/images/train \\
    --labels-dir datasets/homecare/labels/train \\
    --output-dir ./crops_homecare \\
    --classes-file datasets/homecare/classes.txt \\
    --config-file src/data/bbox_crop_config.yaml

  # Crop with padding
  python bbox_cropper.py \\
    --images-dir datasets/homecare/images/train \\
    --labels-dir datasets/homecare/labels/train \\
    --output-dir ./crops_homecare \\
    --classes-file datasets/homecare/classes.txt \\
    --padding 10
        """,
    )

    parser.add_argument(
        "--images-dir",
        type=str,
        required=True,
        help="Directory containing images",
    )
    parser.add_argument(
        "--labels-dir",
        type=str,
        required=True,
        help="Directory containing YOLO format labels",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to save cropped images",
    )
    parser.add_argument(
        "--classes-file",
        type=str,
        required=True,
        help="Path to classes.txt file",
    )
    parser.add_argument(
        "--config-file",
        type=str,
        default=None,
        help="Optional path to bbox_crop_config.yaml for filtering classes",
    )
    parser.add_argument(
        "--padding",
        type=int,
        default=0,
        help="Pixel padding around crops (default: 0)",
    )

    args = parser.parse_args()

    try:
        cropper = BBoxCropper(
            images_dir=args.images_dir,
            labels_dir=args.labels_dir,
            output_dir=args.output_dir,
            classes_file=args.classes_file,
            padding=args.padding,
            config_file=args.config_file,
        )

        cropper.process_folder()
        cropper.print_summary()

    except Exception as e:
        print(f"Error: {e}")
        exit(1)


if __name__ == "__main__":
    main()
