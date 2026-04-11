"""
Inference script for YOLO object detection
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.models import YOLOModel
from src.inference import YOLOPredictor


def main(args):
    """Main inference function"""
    
    print("=" * 50)
    print("YOLO Object Detection Inference")
    print("=" * 50)
    
    # Load model
    print(f"Loading model: {args.weights}")
    model = YOLOModel(device=args.device)
    model.load_weights(args.weights)
    
    # Load class ID mapping if provided
    if args.class_id_map:
        print(f"Loading class ID mapping: {args.class_id_map}")
        model.load_class_id_map(args.class_id_map)
    
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
            print(f"  - Class ID: {det['class_id']}, {det['class_name']}, Confidence: {det['confidence']:.2f}")
    
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
        '--weights',
        type=str,
        required=True,
        help='Path to model weights'
    )
    parser.add_argument(
        '--source',
        type=str,
        required=True,
        help='Source: image/video path or directory'
    )
    parser.add_argument(
        '--mode',
        type=str,
        default='image',
        choices=['image', 'video', 'webcam'],
        help='Inference mode'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output path for video/webcam mode'
    )
    parser.add_argument(
        '--conf',
        type=float,
        default=0.5,
        help='Confidence threshold'
    )
    parser.add_argument(
        '--iou',
        type=float,
        default=0.45,
        help='IOU threshold for NMS'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cpu',
        choices=['cuda', 'cpu'],
        help='Device to use'
    )
    parser.add_argument(
        '--duration',
        type=int,
        default=30,
        help='Duration for webcam mode in seconds'
    )
    parser.add_argument(
        '--fps',
        type=int,
        default=30,
        help='FPS for video output'
    )
    parser.add_argument(
        '--class-id-map',
        type=str,
        default=None,
        help='Path to YAML file with class ID mapping (e.g., configs/convert_class_id_map.yaml)'
    )
    
    args = parser.parse_args()
    main(args)
