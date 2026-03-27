"""
YOLO Inference and Prediction Module
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Union, List, Tuple
import torch


class YOLOPredictor:
    """
    High-level inference interface for YOLO object detection.
    Handles predictions on images, videos, and real-time streams.
    """
    
    def __init__(self, model, conf_threshold=0.5, iou_threshold=0.45):
        """
        Initialize predictor.
        
        Args:
            model: YOLOModel instance
            conf_threshold (float): Confidence threshold
            iou_threshold (float): IOU threshold for NMS
        """
        self.model = model
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
    
    def predict_image(self, image_path: str):
        """
        Predict on single image.
        
        Args:
            image_path (str): Path to image file
        
        Returns:
            dict: Predictions with boxes, classes, and confidences
        """
        results = self.model.predict(
            source=image_path,
            conf=self.conf_threshold,
            iou=self.iou_threshold
        )
        
        return self._format_results(results[0])

    def save_annotated_image(self, image_path: str, predictions: dict, output_path: str):
        """Save an annotated image with predictions drawn on it."""
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Cannot read image: {image_path}")

        annotated = self._draw_boxes(image, predictions)
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(out_path), annotated):
            raise ValueError(f"Failed to write annotated image: {output_path}")

    def save_yolo_txt(self, predictions: dict, output_path: str, save_conf: bool = False):
        """Save predictions into YOLO txt format.

        Detect format:
            class x_center y_center width height [conf]
        OBB format:
            class x1 y1 x2 y2 x3 y3 x4 y4 [conf]
        """
        lines = []
        task = predictions.get('task', 'detect')

        for detection in predictions.get('detections', []):
            cls_id = detection.get('class_id', -1)
            if cls_id < 0:
                continue

            if task == 'obb':
                polygon = detection.get('polygon_normalized', [])
                if len(polygon) < 4:
                    continue

                coords = []
                for point in polygon[:4]:
                    if len(point) != 2:
                        continue
                    coords.append(f"{float(point[0]):.6f}")
                    coords.append(f"{float(point[1]):.6f}")

                if len(coords) != 8:
                    continue

                row = [str(cls_id), *coords]
            else:
                bbox = detection.get('bbox_normalized', {})
                required_keys = ['x_center', 'y_center', 'width', 'height']
                if not all(key in bbox for key in required_keys):
                    continue

                row = [
                    str(cls_id),
                    f"{float(bbox['x_center']):.6f}",
                    f"{float(bbox['y_center']):.6f}",
                    f"{float(bbox['width']):.6f}",
                    f"{float(bbox['height']):.6f}",
                ]

            if save_conf:
                row.append(f"{float(detection.get('confidence', 0.0)):.6f}")

            lines.append(' '.join(row))

        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text('\n'.join(lines) + ('\n' if lines else ''), encoding='utf-8')
    
    def predict_batch(self, image_paths: List[str]):
        """
        Predict on multiple images.
        
        Args:
            image_paths (list): List of image paths
        
        Returns:
            list: List of predictions
        """
        results = self.model.predict(
            source=image_paths,
            conf=self.conf_threshold,
            iou=self.iou_threshold
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
            cv2.imshow('YOLO Detection', frame_with_boxes)
            
            if out:
                out.write(frame_with_boxes)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        if out:
            out.release()
        cv2.destroyAllWindows()
        
        return predictions
    
    @staticmethod
    def _resolve_class_name(names, cls_idx):
        """Resolve class name from Ultralytics names dict/list."""
        if cls_idx < 0:
            return 'Unknown'
        if isinstance(names, dict):
            return names.get(cls_idx, 'Unknown')
        if isinstance(names, list) and cls_idx < len(names):
            return names[cls_idx]
        return 'Unknown'

    @staticmethod
    def _format_results(result):
        """Format YOLO results into standardized dict"""
        detections = []
        task = 'obb' if getattr(result, 'obb', None) is not None else 'detect'

        if task == 'obb' and result.obb is not None:
            for obb in result.obb:
                cls_idx = int(obb.cls[0]) if obb.cls is not None else -1
                polygon = obb.xyxyxyxy[0].tolist() if getattr(obb, 'xyxyxyxy', None) is not None else []

                detection = {
                    'task': 'obb',
                    'class_id': cls_idx,
                    'class_name': YOLOPredictor._resolve_class_name(result.names, cls_idx),
                    'confidence': float(obb.conf[0]) if obb.conf is not None else 0,
                    'polygon': polygon,
                    'polygon_normalized': (
                        obb.xyxyxyxyn[0].tolist()
                        if getattr(obb, 'xyxyxyxyn', None) is not None
                        else []
                    ),
                    'xywhr': (
                        obb.xywhr[0].tolist()
                        if getattr(obb, 'xywhr', None) is not None
                        else []
                    )
                }
                detections.append(detection)
        
        if task == 'detect' and result.boxes is not None:
            for box in result.boxes:
                cls_idx = int(box.cls[0]) if box.cls is not None else -1
                detection = {
                    'task': 'detect',
                    'class_id': cls_idx,
                    'class_name': YOLOPredictor._resolve_class_name(result.names, cls_idx),
                    'confidence': float(box.conf[0]) if box.conf is not None else 0,
                    'bbox': {
                        'x1': float(box.xyxy[0][0]),
                        'y1': float(box.xyxy[0][1]),
                        'x2': float(box.xyxy[0][2]),
                        'y2': float(box.xyxy[0][3])
                    },
                    'bbox_normalized': {
                        'x_center': float(box.xywhn[0][0]),
                        'y_center': float(box.xywhn[0][1]),
                        'width': float(box.xywhn[0][2]),
                        'height': float(box.xywhn[0][3])
                    }
                }
                detections.append(detection)
        
        return {
            'task': task,
            'image_path': result.path if hasattr(result, 'path') else None,
            'detections': detections,
            'image_shape': result.orig_shape if hasattr(result, 'orig_shape') else None
        }
    
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
            class_id = detection['class_id']
            color = colors.get(class_id % 6, (255, 255, 255))

            if detection.get('task') == 'obb' and detection.get('polygon'):
                points = np.array(detection['polygon'], dtype=np.int32)
                if points.ndim == 2 and points.shape[0] >= 4:
                    cv2.polylines(frame_copy, [points], isClosed=True, color=color, thickness=thickness)
                    label_anchor = tuple(points[0])
                else:
                    label_anchor = (10, 20)
            else:
                bbox = detection['bbox']
                x1, y1, x2, y2 = int(bbox['x1']), int(bbox['y1']), int(bbox['x2']), int(bbox['y2'])
                cv2.rectangle(frame_copy, (x1, y1), (x2, y2), color, thickness)
                label_anchor = (x1, y1)

            # Draw label
            label = f"{detection['class_name']} {detection['confidence']:.2f}"
            cv2.putText(
                frame_copy,
                label,
                (int(label_anchor[0]), int(label_anchor[1]) - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                color,
                2
            )
        
        return frame_copy
