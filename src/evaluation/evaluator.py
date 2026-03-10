"""
YOLO Model Evaluation and Metrics
"""

import json
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from collections import defaultdict


class YOLOEvaluator:
    """
    Evaluate YOLO model performance on test set.
    Computes standard object detection metrics: mAP, precision, recall, etc.
    """
    
    def __init__(self, model):
        """
        Initialize evaluator.
        
        Args:
            model: YOLOModel instance
        """
        self.model = model
        self.results = {}
    
    def evaluate(self, test_dir, conf_threshold=0.5, iou_threshold=0.5):
        """
        Evaluate model on test dataset.
        
        Args:
            test_dir (str): Directory containing test images
            conf_threshold (float): Confidence threshold for predictions
            iou_threshold (float): IOU threshold for evaluation
        
        Returns:
            dict: Evaluation metrics
        """
        predictions = self.model.predict(
            source=test_dir,
            conf=conf_threshold
        )
        
        # Process results
        metrics = {
            'total_images': len(predictions),
            'predictions': []
        }
        
        for result in predictions:
            pred_data = {
                'image': result.path,
                'detections': []
            }
            
            if result.boxes is not None:
                for box in result.boxes:
                    pred_data['detections'].append({
                        'class': int(box.cls[0]),
                        'confidence': float(box.conf[0]),
                        'bbox': box.xyxy[0].tolist()
                    })
            
            metrics['predictions'].append(pred_data)
        
        self.results = metrics
        return metrics
    
    def compute_metrics(self, predictions, ground_truths, iou_threshold=0.5):
        """
        Compute precision, recall, and mAP.
        
        Args:
            predictions (list): List of predictions
            ground_truths (list): List of ground truth annotations
            iou_threshold (float): IOU threshold
        
        Returns:
            dict: Computed metrics
        """
        tp = 0
        fp = 0
        fn = 0
        
        for pred, gt in zip(predictions, ground_truths):
            pred_boxes = pred.get('boxes', [])
            gt_boxes = gt.get('boxes', [])
            
            # Match predictions to ground truths
            matched_gt = set()
            
            for pred_box in pred_boxes:
                best_iou = 0
                best_gt_idx = -1
                
                for gt_idx, gt_box in enumerate(gt_boxes):
                    if gt_idx in matched_gt:
                        continue
                    
                    iou = self._compute_iou(pred_box, gt_box)
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = gt_idx
                
                if best_iou >= iou_threshold:
                    tp += 1
                    matched_gt.add(best_gt_idx)
                else:
                    fp += 1
            
            fn += len(gt_boxes) - len(matched_gt)
        
        # Compute metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'tp': tp,
            'fp': fp,
            'fn': fn
        }
    
    @staticmethod
    def _compute_iou(box1, box2):
        """
        Compute IOU between two bounding boxes.
        
        Args:
            box1, box2: Bounding boxes in format [x1, y1, x2, y2]
        
        Returns:
            float: IOU value
        """
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        # Intersection area
        xi1 = max(x1_1, x1_2)
        yi1 = max(y1_1, y1_2)
        xi2 = min(x2_1, x2_2)
        yi2 = min(y2_1, y2_2)
        
        inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        
        # Union area
        box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
        box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = box1_area + box2_area - inter_area
        
        iou = inter_area / union_area if union_area > 0 else 0
        return iou
    
    def save_results(self, save_path):
        """
        Save evaluation results to JSON.
        
        Args:
            save_path (str): Path to save results
        """
        with open(save_path, 'w') as f:
            json.dump(self.results, f, indent=2)
    
    def plot_results(self, save_path=None):
        """
        Plot evaluation results.
        
        Args:
            save_path (str): Path to save plot
        """
        # This is a placeholder for visualization
        # Can be extended with confusion matrices, precision-recall curves, etc.
        pass
