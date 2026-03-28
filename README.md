# Vizon Backend - YOLO Object Detection Training Framework

A comprehensive Python framework for training, evaluating, and deploying Ultralytics YOLOv8 models for object detection.

## Features

- **Easy-to-use training pipeline** with configurable hyperparameters
- **Data loading utilities** supporting YOLO format annotations
- **Model management** with support for different YOLOv8 sizes (n, s, m, l, x)
- **Inference capabilities** on images, videos, and webcam feeds
- **Evaluation metrics** including precision, recall, and mAP
- **Modular architecture** for easy customization and extension

## Project Structure

```
vizon_backend/
├── src/
│   ├── data/              # Data loading and preprocessing
│   │   ├── dataset.py     # Custom YOLO dataset class
│   │   └── dataloader.py  # DataLoader creation utilities
│   ├── models/            # Model definitions
│   │   └── yolo_model.py  # YOLO model wrapper
│   ├── training/          # Training pipeline
│   │   └── trainer.py     # Training manager
│   ├── evaluation/        # Evaluation utilities
│   │   └── evaluator.py   # Model evaluator
│   └── inference/         # Inference utilities
│       └── predictor.py   # Prediction interface
├── configs/               # Configuration files
│   ├── training_config.yaml    # Training parameters
│   └── dataset_config.yaml     # Dataset configuration
├── utils/                 # Helper functions
│   └── helpers.py         # Utility functions
├── datasets/              # Data directory (to be populated)
├── logs/                  # Training logs
├── runs/                  # Training outputs
├── train.py              # Main training script
├── infer.py              # Inference script
├── evaluate.py           # Evaluation script
├── examples.py           # Usage examples
├── convert_obb_labels.py # Label format converter CLI (xywhr -> OBB corners)
└── requirements.txt      # Python dependencies
```

## Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd vizon_backend
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

## Quick Start

### 1. Prepare Your Dataset

Organize your dataset in YOLO format:
```
datasets/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
└── labels/
    ├── train/
    ├── val/
    └── test/
```

Each image in `labels/` should have a corresponding `.txt` file with annotations in YOLO format:
```
<class_id> <x_center> <y_center> <width> <height>
```
(All coordinates normalized to [0, 1])

### 2. Configure Dataset

Edit `configs/dataset_config.yaml`:
```yaml
path: datasets/my_dataset
train: images/train
val: images/val
test: images/test
nc: 5  # Number of classes
names: ['class1', 'class2', 'class3', 'class4', 'class5']
```

### 3. Train Model

```bash
python train.py \
    --data configs/dataset_config.yaml \
    --task detect \
    --epochs 100 \
    --batch-size 32 \
    --imgsz 640 \
    --validate
```

**Train an OBB model:**
```bash
python train.py \
    --data configs/dataset_config.yaml \
    --task obb \
    --model-name yolo26m-obb.pt \
    --epochs 100
```

### 4. Run Inference

**On a single image:**
```bash
python infer.py \
    --weights runs/detect/train/weights/best.pt \
    --task detect \
    --source path/to/image.jpg \
    --mode image
```

**OBB inference:**
```bash
python infer.py \
    --weights runs/run/weights/best.pt \
    --task obb \
    --source path/to/image.jpg \
    --mode image
```

**Save annotated image + YOLO txt predictions (image mode):**
```bash
python infer.py \
    --weights runs/run7/weights/best.pt \
    --task obb \
    --source path/to/image.jpg \
    --mode image \
    --output runs/infer/image_annotated.jpg \
    --save-txt \
    --txt-output runs/infer/image.txt
```

Notes:
- `--output` saves annotated image in `image` mode.
- `--save-txt` saves prediction labels in YOLO format.
- `--txt-output` is optional (default uses source/output stem + `.txt`).
- `--save-conf` appends confidence score to each txt row.

**On a video:**
```bash
python infer.py \
    --weights runs/detect/train/weights/best.pt \
    --source path/to/video.mp4 \
    --mode video \
    --output output.mp4
```

**On webcam:**
```bash
python infer.py \
    --weights runs/detect/train/weights/best.pt \
    --mode webcam \
    --duration 30 \
    --output webcam_output.mp4
```

### 5. Evaluate Model

```bash
python evaluate.py \
    --weights runs/detect/train/weights/best.pt \
    --task detect \
    --test-dir datasets/my_dataset/images/test \
    --output eval_results.json
```

### 6. Convert Labels to YOLO OBB Corner Format

Convert labels from:
```text
class_id x y w h r
```
to:
```text
class_id x1 y1 x2 y2 x3 y3 x4 y4
```

CLI usage:
```bash
python convert_obb_labels.py \
    --input-dir datasets/homecare/labels/train
```

If your angle `r` is in degrees:
```bash
python convert_obb_labels.py \
    --input-dir datasets/homecare/labels/train \
    --output-dir datasets/homecare/labels/train_obb \
    --angle-unit degrees
```

## Usage Examples

### Python API

```python
from src.models import YOLOModel
from src.training import YOLOTrainer
from src.inference import YOLOPredictor
from utils import setup_seed, get_device

# Setup
setup_seed(42)
device = get_device()

# Initialize trainer
trainer = YOLOTrainer(task='detect')

# Train
results = trainer.train(
    data_yaml='configs/dataset_config.yaml',
    epochs=100,
    batch_size=32
)

# Inference
model = YOLOModel(model_size='m', task='detect')
predictor = YOLOPredictor(model)
results = predictor.predict_image('path/to/image.jpg')

# Access detections
for detection in results['detections']:
    print(f"{detection['class_name']}: {detection['confidence']:.2f}")
```

## Configuration

### Training Configuration (`configs/training_config.yaml`)

Key parameters:
- `model_size`: Model size (n/s/m/l/x)
- `task`: YOLO task (`detect` or `obb`)
- `model_name`: Optional explicit checkpoint filename/path
- `epochs`: Number of training epochs
- `batch_size`: Batch size
- `imgsz`: Input image size
- `lr0`: Initial learning rate
- `momentum`: SGD momentum
- `weight_decay`: L2 regularization
- `patience`: Early stopping patience

### Dataset Configuration (`configs/dataset_config.yaml`)

- `path`: Root dataset directory
- `train/val/test`: Paths to train/val/test splits
- `nc`: Number of classes
- `names`: Class names list

## API Reference

### YOLOModel
```python
model = YOLOModel(model_size='m', device='cuda', task='detect')
model.train(data_yaml, epochs=100, batch_size=32)
results = model.predict(source='image.jpg', conf=0.5)
model.export(format='onnx')
```

### YOLOTrainer
```python
trainer = YOLOTrainer(config_path='configs/training_config.yaml')
results = trainer.train(data_yaml, epochs=100)
trainer.validate(data_yaml)
trainer.log_metrics({'loss': 0.5})
```

### YOLOPredictor
```python
predictor = YOLOPredictor(model, conf_threshold=0.5)
results = predictor.predict_image('image.jpg')
predictor.save_annotated_image('image.jpg', results, 'annotated.jpg')
predictor.save_yolo_txt(results, 'predictions.txt', save_conf=False)
results = predictor.predict_video('video.mp4', output_path='output.mp4')
results = predictor.predict_webcam(duration=30)
```

### Label Converter Utilities
```python
from src.data import convert_line_xywhr_to_obb, convert_folder_xywhr_to_obb

line = convert_line_xywhr_to_obb('0 0.5 0.5 0.2 0.1 0.785398', angle_unit='radians')
converted = convert_folder_xywhr_to_obb('labels_in', output_dir='labels_out')
```

### YOLOEvaluator
```python
evaluator = YOLOEvaluator(model)
results = evaluator.evaluate('test_dir/', conf_threshold=0.5)
metrics = evaluator.compute_metrics(predictions, ground_truths)
```

## Advanced Features

### Custom Data Loading
```python
from src.data import YOLODataset, create_dataloader

dataset = YOLODataset(
    img_dir='images/train',
    label_dir='labels/train',
    img_size=640,
    augment=True
)

loader = create_dataloader(
    img_dir='images/train',
    label_dir='labels/train',
    batch_size=32,
    shuffle=True
)
```

### Reproducibility
```python
from utils import setup_seed

# Set random seed for reproducible results
setup_seed(42)
```

## Troubleshooting

### GPU Not Detected
```python
from utils import print_gpu_info
print_gpu_info()
```

### Memory Issues
- Reduce `batch_size`
- Reduce `imgsz`
- Use a smaller model size (n or s instead of l or x)

### Dataset Issues
Ensure your dataset is in correct YOLO format:
- Images: `.jpg`, `.jpeg`, `.png`
- Labels: `.txt` files with same names as images
- Coordinates: Normalized (0-1)

<!-- ...existing code... -->

## Label conversion

The project includes utilities in `src/data/label_converter.py` for converting labels from rotated box format:

- Input format: `class_id x y w h r`

Supported outputs:

1. **YOLO OBB format**
   - Output: `class_id x1 y1 x2 y2 x3 y3 x4 y4`
   - Uses the rotation value `r`

2. **Standard YOLO detection format**
   - Output: `class_id x y w h`
   - Drops the rotation value `r`

### Available functions

- `convert_line_xywhr_to_obb(line, angle_unit="radians", precision=6)`
- `convert_file_xywhr_to_obb(input_path, output_path=None, angle_unit="radians", precision=6)`
- `convert_folder_xywhr_to_obb(input_dir, output_dir=None, angle_unit="radians", precision=6, pattern="*.txt")`

- `convert_line_xywhr_to_xywh(line, precision=6)`
- `convert_file_xywhr_to_xywh(input_path, output_path=None, precision=6)`
- `convert_folder_xywhr_to_xywh(input_dir, output_dir=None, precision=6, pattern="*.txt")`

### Notes

- `x`, `y`, `w`, and `h` are expected to already be YOLO-normalized.
- `r` is only used for OBB conversion.
- If `output_path` or `output_dir` is not provided, conversion is done in-place.

### Example

````python
from pathlib import Path

from src.data.label_converter import (
    convert_file_xywhr_to_obb,
    convert_file_xywhr_to_xywh,
)

convert_file_xywhr_to_obb(
    input_path=Path("labels/sample.txt"),
    output_path=Path("labels_obb/sample.txt"),
    angle_unit="radians",
)

convert_file_xywhr_to_xywh(
    input_path=Path("labels/sample.txt"),
    output_path=Path("labels_yolo/sample.txt"),
)
````

## Future Improvements

- [ ] Multi-GPU training support
- [ ] Mixed precision training
- [ ] Advanced augmentation strategies
- [ ] Model ensemble support
- [ ] TensorFlow/PyTorch conversion utilities
- [ ] Web UI for training monitoring
- [ ] Docker containerization

## References

- [Ultralytics YOLOv8](https://docs.ultralytics.com/)
- [YOLOv8 GitHub](https://github.com/ultralytics/ultralytics)

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Support

For issues and questions, please open an issue on the GitHub repository.
