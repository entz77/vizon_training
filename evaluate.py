"""
Evaluation script for YOLO model
"""

import argparse
import sys
from pathlib import Path
import json

sys.path.insert(0, str(Path(__file__).parent))

from src.models import YOLOModel
from src.evaluation import YOLOEvaluator


def main(args):
    """Main evaluation function"""
    
    print("=" * 50)
    print("YOLO Model Evaluation")
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
    
    # Initialize evaluator
    evaluator = YOLOEvaluator(model)
    
    # Evaluate
    print(f"Evaluating on: {args.test_dir}")
    results = evaluator.evaluate(
        test_dir=args.test_dir,
        conf_threshold=args.conf,
        iou_threshold=args.iou
    )
    
    print(f"\nTotal images evaluated: {results['total_images']}")
    
    # Save results
    if args.output:
        evaluator.save_results(args.output)
        print(f"Results saved to: {args.output}")
    
    print("=" * 50)
    print("Evaluation completed!")
    print("=" * 50)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Evaluate YOLO model'
    )
    
    parser.add_argument(
        '--weights',
        type=str,
        required=True,
        help='Path to model weights'
    )
    parser.add_argument(
        '--task',
        type=str,
        default='detect',
        choices=['detect', 'obb'],
        help='YOLO task type'
    )
    parser.add_argument(
        '--model-size',
        type=str,
        default='m',
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
        '--test-dir',
        type=str,
        required=True,
        help='Path to test images directory'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='results.json',
        help='Output path for results JSON'
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
        default=0.5,
        help='IOU threshold'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        choices=['cuda', 'cpu'],
        help='Device to use'
    )
    
    args = parser.parse_args()
    main(args)
