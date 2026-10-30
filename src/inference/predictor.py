"""
Object Detection Inference and Prediction Module
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Union, List, Tuple, Optional, Dict
import torch


class DetectorPredictor:
    """
    High-level inference interface for object detection.
    Handles predictions on images, videos, and real-time streams.
    """
    
    def __init__(self, model, conf_threshold=0.5, iou_threshold=0.45, class_id_map: Optional[Dict[int, int]] = None):
        """
        Initialize predictor.
        
        Args:
            model: BaseDetector instance (YOLOModel, RFDETRModel, ...)
            conf_threshold (float): Confidence threshold
            iou_threshold (float): IOU threshold for NMS
            class_id_map (dict): Optional mapping of class IDs. Maps original class IDs to new class IDs.
                                 If not provided, uses the model's class_id_map if available.
        """
        self.model = model
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        # Use provided mapping or fall back to model's mapping
        self.class_id_map = class_id_map if class_id_map is not None else getattr(model, 'class_id_map', None)
    
    def predict_image(self, image_path: str, verbose: bool = True):
        """
        Predict on single image.

        Args:
            image_path (str): Path to image file
            verbose (bool): Print per-image inference logs

        Returns:
            dict: Predictions with boxes, classes, and confidences
        """
        results = self.model.predict(
            source=image_path,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            verbose=verbose
        )

        return self._format_results(results[0])
    
    def predict_batch(self, image_paths: List[str], verbose: bool = True):
        """
        Predict on multiple images.
        
        Args:
            image_paths (list): List of image paths
            verbose (bool): Print per-image inference logs
        
        Returns:
            list: List of predictions
        """
        results = self.model.predict(
            source=image_paths,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            verbose=verbose
        )
        
        return [self._format_results(r) for r in results]
    
    def predict_video(self, video_path: str, output_path: str = None, fps: int = 30):
        """
        Predict on video file.
        
        Args:
            video_path (str): Path to video file
            output_path (str): Path to save output video (optional)
            fps (int): Frame processing speed
        
        Returns:
            list: Predictions for each frame
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Setup video writer if output path specified
        out = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
        
        predictions = []
        frame_idx = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Predict on frame
            result = self.model.predict(
                source=frame,
                conf=self.conf_threshold,
                iou=self.iou_threshold
            )[0]
            
            pred_data = self._format_results(result)
            predictions.append(pred_data)
            
            # Draw predictions on frame
            frame_with_boxes = self._draw_boxes(frame, pred_data)
            
            if out:
                out.write(frame_with_boxes)
            
            frame_idx += 1
            print(f"Processed frame {frame_idx}/{total_frames}")
        
        cap.release()
        if out:
            out.release()
        
        return predictions
    
    def predict_webcam(self, duration: int = 30, output_path: str = None):
        """
        Predict on webcam feed.
        
        Args:
            duration (int): Duration in seconds
            output_path (str): Optional path to save video
        
        Returns:
            list: Predictions
        """
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            raise ValueError("Cannot open webcam")
        
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        out = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, 30, (frame_width, frame_height))
        
        predictions = []
        start_time = cv2.getTickCount()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Check elapsed time
            elapsed = (cv2.getTickCount() - start_time) / cv2.getTickFrequency()
            if elapsed > duration:
                break
            
            # Predict
            result = self.model.predict(
                source=frame,
                conf=self.conf_threshold,
                iou=self.iou_threshold
            )[0]
            
            pred_data = self._format_results(result)
            predictions.append(pred_data)
            
            # Draw and display
            frame_with_boxes = self._draw_boxes(frame, pred_data)
            cv2.imshow('Detection', frame_with_boxes)
            
            if out:
                out.write(frame_with_boxes)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        if out:
            out.release()
        cv2.destroyAllWindows()
        
        return predictions
    
    def _format_results(self, result):
        """Format backend-neutral ImageDetections into a standardized dict, applying class ID mapping if available"""
        detections = []
        shape = result.image_shape
        h, w = shape if shape is not None else (None, None)

        for det in result.detections:
            original_class_id = det.class_id
            # Apply class ID mapping if available
            mapped_class_id = original_class_id
            if self.class_id_map is not None and original_class_id in self.class_id_map:
                mapped_class_id = self.class_id_map[original_class_id]

            detection = {
                'class_id': mapped_class_id,
                'original_class_id': original_class_id if original_class_id != mapped_class_id else None,
                'class_name': det.class_name,
                'confidence': det.confidence,
                'bbox': {'x1': det.x1, 'y1': det.y1, 'x2': det.x2, 'y2': det.y2},
            }
            if w and h:
                detection['bbox_normalized'] = {
                    'x_center': (det.x1 + det.x2) / 2 / w,
                    'y_center': (det.y1 + det.y2) / 2 / h,
                    'width': (det.x2 - det.x1) / w,
                    'height': (det.y2 - det.y1) / h,
                }
            detections.append(detection)

        return {
            'image_path': result.image_path,
            'detections': detections,
            'image_shape': shape
        }
    
    def load_class_id_map(self, yaml_path):
        """
        Load class ID mapping from a YAML configuration file.
        
        Args:
            yaml_path (str or Path): Path to the YAML mapping file
            
        Example YAML format:
            class_id_map:
              0: 2
              1: 1
              2: 0
        
        Returns:
            dict: The loaded class ID mapping
        """
        from src.data.label_converter import load_class_id_map_from_yaml
        self.class_id_map = load_class_id_map_from_yaml(Path(yaml_path))
        return self.class_id_map
    
    def save_annotated(self, image_path, predictions, output_path):
        """
        Draw `predictions` (as returned by predict_image) on the image and save it.

        Args:
            image_path (str): Source image
            predictions (dict): Result of predict_image
            output_path (str): Output file, or a directory (the source file name is reused)

        Returns:
            Path: Where the image was written
        """
        out = Path(output_path)
        if out.is_dir() or not out.suffix:
            out = out / Path(image_path).name
        out.parent.mkdir(parents=True, exist_ok=True)
        frame = cv2.imread(str(image_path))
        if frame is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")
        cv2.imwrite(str(out), self._draw_boxes(frame, predictions))
        return out

    @staticmethod
    def _draw_boxes(frame, predictions, thickness=2, font_scale=0.6):
        """
        Draw bounding boxes on frame.
        
        Args:
            frame: Input frame
            predictions: Prediction data
            thickness: Box line thickness
            font_scale: Text font scale
        
        Returns:
            frame: Frame with drawn boxes
        """
        frame_copy = frame.copy()
        
        # Define colors for different classes
        colors = {
            0: (255, 0, 0),      # Blue
            1: (0, 255, 0),      # Green
            2: (0, 0, 255),      # Red
            3: (255, 255, 0),    # Cyan
            4: (255, 0, 255),    # Magenta
            5: (0, 255, 255)     # Yellow
        }
        
        for detection in predictions['detections']:
            bbox = detection['bbox']
            x1, y1, x2, y2 = int(bbox['x1']), int(bbox['y1']), int(bbox['x2']), int(bbox['y2'])
            
            class_id = detection['class_id']
            color = colors.get(class_id % 6, (255, 255, 255))
            
            # Draw rectangle
            cv2.rectangle(frame_copy, (x1, y1), (x2, y2), color, thickness)
            
            # Draw label
            label = f"{detection['class_name']} {detection['confidence']:.2f}"
            cv2.putText(
                frame_copy,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                color,
                2
            )
        
        return frame_copy


# Backwards-compatible alias
YOLOPredictor = DetectorPredictor
