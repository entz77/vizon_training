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

### 2. Label Conversion Utilities

The framework provides utilities to convert between different label formats:

#### Convert XYWHR to OBB Format (Rotated Bboxes)
Convert from `class_id x y w h r` (rotated bounding box) to YOLO OBB format:

**CLI:**
```bash
python -c "from src.data.label_converter import convert_folder_xywhr_to_obb; \
convert_folder_xywhr_to_obb('input_labels', 'output_labels')"
```

**Python API:**
```python
from src.data.label_converter import convert_folder_xywhr_to_obb, convert_file_xywhr_to_obb

# Convert entire folder
results = convert_folder_xywhr_to_obb(
    input_dir='datasets/labels_rotated',
    output_dir='datasets/labels_obb',
    angle_unit='radians',  # or 'degrees'
    precision=6
)

# Convert single file
convert_file_xywhr_to_obb(
    input_path='labels/image1.txt',
    output_path='labels_obb/image1.txt'
)
```

#### Convert XYWHR to Standard YOLO XYWH Format
Convert from `class_id x y w h r` to standard YOLO `class_id x y w h` format:

**CLI:**
```bash
python convert_xywhr_to_xywh.py --input-dir datasets/labels_rotated --output-dir datasets/labels
python convert_xywhr_to_xywh.py --input-dir in_labels --output-dir out_labels --precision 6
```

**Python API:**
```python
from src.data.label_converter import convert_folder_xywhr_to_xywh, convert_file_xywhr_to_xywh

# Convert entire folder
results = convert_folder_xywhr_to_xywh(
    input_dir='datasets/labels_rotated',
    output_dir='datasets/labels',
    precision=6
)

# Convert single file
convert_file_xywhr_to_xywh(
    input_path='labels/image1.txt',
    output_path='labels/image1_converted.txt',
    precision=6
)
```

#### Convert Class IDs to 0
Convert all class IDs in label files to 0 (useful for single-class models or data preparation):

**CLI:**
```bash
python convert_class_id_to_zero.py --input-dir datasets/labels
python convert_class_id_to_zero.py --input-dir in_labels --output-dir out_labels
```

**Python API:**
```python
from src.data.label_converter import convert_folder_class_id_to_zero, convert_file_class_id_to_zero

# Convert entire folder (in-place)
results = convert_folder_class_id_to_zero(
    input_dir='datasets/labels'
)

# Convert with output directory
results = convert_folder_class_id_to_zero(
    input_dir='datasets/labels',
    output_dir='datasets/labels_zero_class'
)

# Convert single file
convert_file_class_id_to_zero(
    input_path='labels/image1.txt',
    output_path='labels/image1_zero_class.txt'
)
```

### 3. Configure Dataset

Edit `configs/dataset_config.yaml`:
```yaml
path: datasets/my_dataset
train: images/train
val: images/val
test: images/test
nc: 5  # Number of classes
names: ['class1', 'class2', 'class3', 'class4', 'class5']
```

### 4. Train Model

```bash
python train.py \
    --data configs/dataset_config.yaml \
    --epochs 100 \
    --batch-size 32 \
    --imgsz 640 \
    --validate
```

### 5. Run Inference

**On a single image:**
```bash
python infer.py \
    --weights runs/detect/train/weights/best.pt \
    --source path/to/image.jpg \
    --mode image
```

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

### 6. Evaluate Model

```bash
python evaluate.py \
    --weights runs/detect/train/weights/best.pt \
    --test-dir datasets/my_dataset/images/test \
    --output eval_results.json
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
trainer = YOLOTrainer()

# Train
results = trainer.train(
    data_yaml='configs/dataset_config.yaml',
    epochs=100,
    batch_size=32
)

# Inference
model = YOLOModel(model_size='m')
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
model = YOLOModel(model_size='m', device='cuda')
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
results = predictor.predict_video('video.mp4', output_path='output.mp4')
results = predictor.predict_webcam(duration=30)
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
