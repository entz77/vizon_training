# Getting Started Guide

## Prerequisites

- Python 3.8+
- CUDA 11.8+ (optional but recommended for GPU acceleration)
- pip or conda package manager

## Installation Steps

### 1. Clone Repository
```bash
git clone https://github.com/yourusername/vizon_backend.git
cd vizon_backend
```

### 2. Create Virtual Environment

**Using venv:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

**Using conda:**
```bash
conda create -n yolo python=3.10
conda activate yolo
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

**For development:**
```bash
pip install -r requirements-dev.txt
```

### 4. Verify Installation
```bash
python -c "from src.models import YOLOModel; print('Installation successful!')"
```

## Dataset Preparation

### YOLO Format Requirements

Your dataset should follow this structure:
```
datasets/
├── images/
│   ├── train/
│   │   ├── img1.jpg
│   │   ├── img2.jpg
│   │   └── ...
│   ├── val/
│   │   └── ...
│   └── test/
│       └── ...
└── labels/
    ├── train/
    │   ├── img1.txt
    │   ├── img2.txt
    │   └── ...
    ├── val/
    │   └── ...
    └── test/
        └── ...
```

### Label Format

Each `.txt` file should contain one line per object:
```
<class_id> <x_center> <y_center> <width> <height>
```

**Example `img1.txt`:**
```
0 0.5 0.5 0.3 0.4
1 0.2 0.3 0.1 0.2
```

All coordinates are normalized to [0, 1] range.

### Download Public Datasets

**COCO Dataset:**
```bash
# Installation instructions are available through various sources
# Consider using roboflow or kaggle for easy dataset access
```

## Configuration

### Edit Training Configuration

Open `configs/training_config.yaml`:

```yaml
model_size: 'm'      # Model architecture
epochs: 50           # Number of epochs to train
batch_size: 32       # Batch size (adjust based on GPU memory)
imgsz: 640          # Image size
patience: 10        # Early stopping patience
lr0: 0.01           # Initial learning rate
```

### Edit Dataset Configuration

Open `configs/dataset_config.yaml`:

```yaml
path: datasets/my_dataset
train: images/train
val: images/val
test: images/test
nc: 5                              # Number of classes
names: ['apple', 'banana', 'orange', 'grape', 'kiwi']
```

## Training

### Basic Training
```bash
python train.py
```

### Custom Training
```bash
python train.py \
    --data configs/dataset_config.yaml \
    --epochs 100 \
    --batch-size 32 \
    --imgsz 640 \
    --lr0 0.001 \
    --momentum 0.937 \
    --weight-decay 0.0005 \
    --patience 20 \
    --validate
```

### Resume Training
```bash
python train.py \
    --config runs/run/args.yaml \
    --resume
```

## Model Inference

### Single Image
```bash
python infer.py \
    --weights runs/detect/weights/best.pt \
    --source images/test.jpg \
    --mode image \
    --conf 0.5
```

### Batch Inference
```bash
python infer.py \
    --weights runs/detect/weights/best.pt \
    --source images/test_folder/ \
    --mode image
```

### Video Inference
```bash
python infer.py \
    --weights runs/detect/weights/best.pt \
    --source video.mp4 \
    --mode video \
    --output output.mp4 \
    --fps 30
```

### Real-time Webcam
```bash
python infer.py \
    --weights runs/detect/weights/best.pt \
    --mode webcam \
    --duration 60 \
    --output webcam_output.mp4
```

## Model Evaluation

```bash
python evaluate.py \
    --weights runs/detect/weights/best.pt \
    --test-dir datasets/my_dataset/images/test \
    --output results.json \
    --conf 0.5
```

## Python API Usage

### Load and Train
```python
from src.training import YOLOTrainer
from utils import setup_seed, get_device

# Setup
setup_seed(42)

# Create trainer
trainer = YOLOTrainer(config_path='configs/training_config.yaml')

# Train
results = trainer.train(
    data_yaml='configs/dataset_config.yaml',
    epochs=50,
    batch_size=32,
    imgsz=640
)
```

### Make Predictions
```python
from src.models import YOLOModel
from src.inference import YOLOPredictor

# Load model
model = YOLOModel(model_size='m')
model.load_weights('runs/detect/train/weights/best.pt')

# Create predictor
predictor = YOLOPredictor(model, conf_threshold=0.5)

# Predict
results = predictor.predict_image('test.jpg')
print(f"Found {len(results['detections'])} objects")

for det in results['detections']:
    print(f"{det['class_name']}: {det['confidence']:.2f}")
```

### Batch Processing
```python
from pathlib import Path

image_paths = list(Path('test_images').glob('*.jpg'))
results = predictor.predict_batch(image_paths)

print(f"Processed {len(results)} images")
```

## GPU Configuration

### Check GPU Availability
```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"CUDA version: {torch.version.cuda}")
```

### Force CPU Usage
```bash
python train.py --no-cuda
```

## Troubleshooting

### OutOfMemory (OOM) Error
```bash
# Reduce batch size
python train.py --batch-size 16

# Reduce image size
python train.py --imgsz 416

# Use smaller model
python train.py --model_size n  # Use nano instead of medium
```

### Slow Training
- Ensure GPU is being used: `nvidia-smi`
- Verify CUDA installation
- Check number of workers in dataloader
- Reduce image augmentation

### Data Loading Issues
- Verify label and image file names match (except extension)
- Check YAML paths are relative or absolute correctly
- Ensure coordinates are normalized to [0, 1]

### Module Import Errors
```bash
# Add current directory to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
python train.py
```

## Performance Tips

1. **Use appropriate batch size** for your GPU:
   - V100 16GB: batch_size=64
   - RTX 3090: batch_size=128
   - T4 16GB: batch_size=32

2. **Image size vs speed trade-off**:
   - 640: Best accuracy, slower
   - 416: Good balance
   - 320: Faster, lower accuracy

3. **Model size selection**:
   - Nano (n): ~5M params, fastest
   - Small (s): ~11M params
   - Medium (m): ~25M params
   - Large (l): ~53M params
   - XLarge (x): ~71M params

4. **Data augmentation**:
   - More augmentation → better generalization but slower training
   - Adjust in `training_config.yaml`

## Next Steps

- Explore [examples.py](examples.py) for more examples
- Read [README.md](README.md) for API reference
- Check [Ultralytics documentation](https://docs.ultralytics.com/)

## Common Issues & Solutions

| Issue | Solution |
|-------|----------|
| "No module named 'src'" | Run from project root, ensure PYTHONPATH is set |
| GPU memory error | Reduce batch size or image size |
| Slow data loading | Increase num_workers in dataloader |
| Poor accuracy | More training data, better augmentation, tune hyperparameters |
| Training stuck | Check GPU usage with nvidia-smi |

## Resources

- [YOLO Object Detection](https://en.wikipedia.org/wiki/You_Only_Look_Once)
- [Ultralytics YOLOv8 Docs](https://docs.ultralytics.com/)
- [PyTorch Documentation](https://pytorch.org/docs/)
- [Computer Vision Basics](https://docs.opencv.org/)
