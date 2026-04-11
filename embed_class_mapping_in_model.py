"""
Utility to embed class ID mapping in YOLO model.
This modifies the model's class names to reflect the mapped class IDs.

Usage:
    python embed_class_mapping_in_model.py \
        --model models/best.pt \
        --mapping configs/inference_class_id_map.yaml \
        --output models/best_mapped.pt
"""

import argparse
from pathlib import Path
from ultralytics import YOLO
import yaml


def load_class_id_map(yaml_path: Path) -> dict:
    """Load class ID mapping from YAML file."""
    with open(yaml_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    if not isinstance(config, dict) or "class_id_map" not in config:
        raise ValueError(f"Invalid YAML format. Expected 'class_id_map' key in {yaml_path}")
    
    class_id_map = config["class_id_map"]
    if not isinstance(class_id_map, dict):
        raise ValueError(f"'class_id_map' must be a dictionary in {yaml_path}")
    
    return {int(k): int(v) for k, v in class_id_map.items()}


def embed_mapping_in_model(model_path: str, class_id_map: dict, output_path: str):
    """
    Embed class ID mapping in model by saving it alongside the model file.
    The mapping is automatically loaded when the model is used with the predictor.
    
    Args:
        model_path: Path to original model
        class_id_map: Dictionary mapping original class IDs to new class IDs
        output_path: Path to save modified model
    """
    print(f"Loading model from {model_path}")
    model = YOLO(model_path)
    
    print(f"Original class names: {model.names}")
    print(f"Class ID mapping: {class_id_map}")
    
    # Save model to output path
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"\nCopying model to {output_path}")
    model.save(str(output_path))
    
    # Save mapping file alongside the model with same name
    mapping_file = output_path.parent / f"{output_path.stem}.mapping.yaml"
    print(f"Saving class ID mapping to {mapping_file}")
    
    mapping_content = {
        'class_id_map': class_id_map
    }
    
    with open(mapping_file, 'w', encoding='utf-8') as f:
        yaml.dump(mapping_content, f, default_flow_style=False)
    
    print(f"✓ Model saved to {output_path}")
    print(f"✓ Mapping saved to {mapping_file}")
    print(f"\nMapping details:")
    for original_id, mapped_id in class_id_map.items():
        if original_id in model.names:
            print(f"  Class {original_id} '{model.names[original_id]}' → ID {mapped_id}")


def main():
    parser = argparse.ArgumentParser(
        description="Embed class ID mapping in YOLO model"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to original YOLO model"
    )
    parser.add_argument(
        "--mapping",
        type=str,
        required=True,
        help="Path to YAML file with class ID mapping"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to save model with embedded mapping"
    )
    
    args = parser.parse_args()
    
    # Load mapping
    try:
        class_id_map = load_class_id_map(Path(args.mapping))
    except Exception as e:
        print(f"Error loading mapping: {e}")
        return
    
    # Embed mapping in model
    try:
        embed_mapping_in_model(args.model, class_id_map, args.output)
    except Exception as e:
        print(f"Error embedding mapping: {e}")
        return


if __name__ == "__main__":
    main()
