"""
Inference script for YOLO object detection
"""

import argparse
import sys
from pathlib import Path
import yaml
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

from src.models import YOLOModel
from src.inference import YOLOPredictor


def load_infer_config(config_path):
    """Load inference config YAML and normalize key style."""
    path = Path(config_path)
    if not path.exists():
        return {}

    with open(path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f) or {}

    normalized = {}
    for key, value in config.items():
        normalized[key.replace('-', '_')] = value
    return normalized


def resolve_args(args):
    """Resolve final inference settings from CLI + config file."""
    config = load_infer_config(args.config)

    def pick(cli_value, *config_keys, default=None):
        if cli_value is not None:
            return cli_value
        for key in config_keys:
            if key in config and config[key] is not None:
                return config[key]
        return default

    source = pick(args.source, 'source', 'image_path')
    weights = pick(args.weights, 'weights', 'model_path')
    mode = pick(args.mode, 'mode', default='image')

    if source is None and mode != 'webcam':
        raise ValueError("Source is required (set --source or 'source'/'image_path' in config)")
    if weights is None:
        raise ValueError("Weights are required (set --weights or 'weights'/'model_path' in config)")

    output_dir = Path(pick(args.output_dir, 'output_dir', default='runs/infer'))
    output_dir.mkdir(parents=True, exist_ok=True)

    if mode == 'image':
        source_path = Path(source)
        stem = source_path.stem
        ext = source_path.suffix if source_path.suffix else '.jpg'

        output = pick(args.output, 'output', default=str(output_dir / f"{stem}_annotated{ext}"))

        save_txt = pick(args.save_txt, 'save_txt', default=False)
        txt_output = pick(args.txt_output, 'txt_output')
        if save_txt and txt_output is None:
            txt_output = str(output_dir / f"{stem}.txt")
    elif mode == 'video':
        source_path = Path(source)
        stem = source_path.stem
        ext = source_path.suffix if source_path.suffix else '.mp4'
        output = pick(args.output, 'output', default=str(output_dir / f"{stem}_annotated{ext}"))
        save_txt = pick(args.save_txt, 'save_txt', default=False)
        txt_output = pick(args.txt_output, 'txt_output')
    else:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output = pick(args.output, 'output', default=str(output_dir / f"webcam_{timestamp}.mp4"))
        save_txt = pick(args.save_txt, 'save_txt', default=False)
        txt_output = pick(args.txt_output, 'txt_output')

    return argparse.Namespace(
        config=args.config,
        weights=weights,
        task=pick(args.task, 'task', default='detect'),
        model_size=pick(args.model_size, 'model_size', default='m'),
        model_name=pick(args.model_name, 'model_name', default=None),
        source=source,
        mode=mode,
        output=output,
        output_dir=str(output_dir),
        save_txt=bool(save_txt),
        txt_output=txt_output,
        save_conf=bool(pick(args.save_conf, 'save_conf', default=False)),
        conf=float(pick(args.conf, 'conf', default=0.5)),
        iou=float(pick(args.iou, 'iou', default=0.45)),
        device=pick(args.device, 'device', default='cuda'),
        duration=int(pick(args.duration, 'duration', default=30)),
        fps=int(pick(args.fps, 'fps', default=30)),
    )


def main(args):
    """Main inference function"""
    
    print("=" * 50)
    print("YOLO Object Detection Inference")
    print("=" * 50)
    print(f"Task: {args.task}")
    
    # Load model
    print(f"Loading model: {args.weights}")
    model = YOLOModel(
        model_size=args.model_size,
        device=args.device,
        task=args.task,
        model_name=args.model_name
    )
    model.load_weights(args.weights)
    
    # Initialize predictor
    predictor = YOLOPredictor(
        model=model,
        conf_threshold=args.conf,
        iou_threshold=args.iou
    )
    
    # Run inference based on source type
    if args.mode == 'image':
        print(f"Predicting on image: {args.source}")
        results = predictor.predict_image(args.source)
        print(f"Detections: {len(results['detections'])}")
        
        for det in results['detections']:
            print(f"  - {det['class_name']}: {det['confidence']:.2f}")

        if args.output:
            predictor.save_annotated_image(args.source, results, args.output)
            print(f"Annotated image saved to: {args.output}")

        if args.save_txt:
            if args.txt_output:
                txt_output_path = args.txt_output
            else:
                base_source = Path(args.output) if args.output else Path(args.source)
                txt_output_path = str(base_source.with_suffix('.txt'))

            predictor.save_yolo_txt(results, txt_output_path, save_conf=args.save_conf)
            print(f"YOLO txt output saved to: {txt_output_path}")
    
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
        description='Run inference with YOLO model'
    )

    parser.add_argument(
        '--config',
        type=str,
        default='configs/infer_config.yaml',
        help='Path to inference config YAML file'
    )
    
    parser.add_argument(
        '--weights',
        type=str,
        default=None,
        help='Path to model weights'
    )
    parser.add_argument(
        '--task',
        type=str,
        default=None,
        choices=['detect', 'obb'],
        help='YOLO task type'
    )
    parser.add_argument(
        '--model-size',
        type=str,
        default=None,
        choices=['n', 's', 'm', 'l', 'x'],
        help='Model size for base model initialization'
    )
    parser.add_argument(
        '--model-name',
        type=str,
        default=None,
        help='Optional explicit model checkpoint name/path for base initialization'
    )
    parser.add_argument(
        '--source',
        type=str,
        default=None,
        help='Source: image/video path or directory'
    )
    parser.add_argument(
        '--mode',
        type=str,
        default=None,
        choices=['image', 'video', 'webcam'],
        help='Inference mode'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output path (overrides default path under runs/infer)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory for generated files (default: runs/infer)'
    )
    parser.add_argument(
        '--save-txt',
        action=argparse.BooleanOptionalAction,
        default=None,
        help='Save YOLO-format prediction txt for image mode'
    )
    parser.add_argument(
        '--txt-output',
        type=str,
        default=None,
        help='Optional txt output path for image mode (defaults to source/output stem + .txt)'
    )
    parser.add_argument(
        '--save-conf',
        action=argparse.BooleanOptionalAction,
        default=None,
        help='Include confidence score in saved txt predictions'
    )
    parser.add_argument(
        '--conf',
        type=float,
        default=None,
        help='Confidence threshold'
    )
    parser.add_argument(
        '--iou',
        type=float,
        default=None,
        help='IOU threshold for NMS'
    )
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        choices=['cuda', 'cpu'],
        help='Device to use'
    )
    parser.add_argument(
        '--duration',
        type=int,
        default=None,
        help='Duration for webcam mode in seconds'
    )
    parser.add_argument(
        '--fps',
        type=int,
        default=None,
        help='FPS for video output'
    )
    
    cli_args = parser.parse_args()
    args = resolve_args(cli_args)
    main(args)
