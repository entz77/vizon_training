"""
Evaluation script for detection models
"""

import argparse
import sys
from pathlib import Path
import json

sys.path.insert(0, str(Path(__file__).parent))

from src.models import build_model, available_models
from src.evaluation import DetectorEvaluator, build_report


def main(args):
    """Main evaluation function"""
    
    print("=" * 50)
    print("Model Evaluation")
    print("=" * 50)
    
    # Load model
    print(f"Loading model: {args.weights}")
    model = build_model(args.model_type, args.model_size, device=args.device)
    model.load_weights(args.weights)
    
    if args.data:
        print(f"Building report on '{args.split}' split of {args.data}")
        report = build_report(
            model, args.data, split=args.split, out_dir=args.report_dir,
            conf=args.report_conf, iou=args.iou, num_samples=args.report_samples
        )
        print(f"mAP50-95: {report['map50_95']:.4f} | mAP50: {report['map50']:.4f} | "
              f"best F1: {report['best_f1']:.3f} @ conf {report['best_f1_conf']:.2f}")
        for w in report['warnings']:
            print(f"WARNING: {w}")
        print(f"Report saved to: {args.report_dir}")

    if args.test_dir:
        # Initialize evaluator
        evaluator = DetectorEvaluator(model)

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
        description='Evaluate detection model'
    )
    
    parser.add_argument(
        '--model-type',
        type=str,
        default='yolo',
        choices=available_models(),
        help='Detection backend'
    )
    parser.add_argument(
        '--model-size',
        type=str,
        default=None,
        help="Model size/variant (backend default if omitted, e.g. 'm' for yolo, 'base' for rfdetr)"
    )
    parser.add_argument(
        '--weights',
        type=str,
        required=True,
        help='Path to model weights'
    )
    parser.add_argument(
        '--test-dir',
        type=str,
        default=None,
        help='Path to test images directory (predictions only, no metrics)'
    )
    parser.add_argument(
        '--data',
        type=str,
        default=None,
        help='Dataset config YAML; builds a metrics report (mAP, confusion matrix, PR/F1 curves)'
    )
    parser.add_argument(
        '--split',
        type=str,
        default='val',
        choices=['train', 'val', 'test'],
        help='Dataset split to build the report on'
    )
    parser.add_argument(
        '--report-dir',
        type=str,
        default='report',
        help='Output directory for the report'
    )
    parser.add_argument(
        '--report-conf',
        type=float,
        default=0.25,
        help='Confidence threshold for the confusion matrix and per-class P/R/F1'
    )
    parser.add_argument(
        '--report-samples',
        type=int,
        default=16,
        help='Number of sample images (ground truth vs predictions) to save with the report, 0 to disable'
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
    if not args.data and not args.test_dir:
        parser.error('provide --data (metrics report) and/or --test-dir (predictions)')
    main(args)
