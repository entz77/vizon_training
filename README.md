# Vizon - Object Detection Training Framework

A Python framework for preparing datasets, training, evaluating, and running inference with
object detection models — including tooling to pull annotations and images directly from GCS.

Models are pluggable backends behind a common `BaseDetector` interface (`src/models/`).
Currently available: `yolo` (Ultralytics) and `rfdetr` (RF-DETR, requires `pip install rfdetr`).
Select one with `model_type` / `model_size` in `configs/training_config.yaml`, or `--model-type` /
`--model-size` in `infer.py` and `evaluate.py`. To add a backend, subclass `BaseDetector` in
`src/models/<name>_model.py` and register it in `src/models/registry.py`.

## Table of Contents

1. [Project Structure](#project-structure)
2. [Installation](#installation)
3. [Workflow](#workflow)
   - [1. Prepare a Dataset](#1-prepare-a-dataset)
   - [2. Convert / Fix Labels](#2-convert--fix-labels)
   - [3. Crop Bounding Boxes (optional)](#3-crop-bounding-boxes-optional)
   - [4. Configure the Dataset](#4-configure-the-dataset)
   - [5. Train](#5-train)
   - [6. Run Inference](#6-run-inference)
   - [7. Remap Class IDs at Inference Time](#7-remap-class-ids-at-inference-time)
   - [8. Evaluate](#8-evaluate)
   - [9. Stitch Videos or Photos into Wide Images](#9-stitch-videos-or-photos-into-wide-images)
   - [10. Preprocess Images](#10-preprocess-images-perspective-rectification-crop)
4. [Notebooks](#notebooks)
5. [Python API Reference](#python-api-reference)
6. [Configuration Reference](#configuration-reference)
7. [GPU & Performance](#gpu--performance)
8. [Troubleshooting](#troubleshooting)
9. [Development](#development)
10. [References](#references)

## Project Structure

```
vizon_training/
├── src/
│   ├── data/                        # Data loading, conversion, GCS dataset prep
│   │   ├── dataset.py                 # Custom YOLO PyTorch Dataset
│   │   ├── dataloader.py              # DataLoader creation utilities
│   │   ├── label_converter.py         # Label format conversion (OBB, xywh, class remap)
│   │   ├── preprocessing.py            # GCS-native train/val/test split
│   │   └── bbox_cropper.py             # Crop bounding boxes out of labeled images
│   ├── models/                      # Pluggable detection backends
│   │   ├── base.py                    # BaseDetector interface + Detection/ImageDetections types
│   │   ├── registry.py                # build_model(model_type, model_size) + backend table
│   │   ├── yolo_model.py              # Ultralytics YOLO backend
│   │   └── rfdetr_model.py            # RF-DETR backend
│   ├── training/trainer.py          # Training pipeline + logging (any backend)
│   ├── evaluation/
│   │   ├── evaluator.py               # Metrics (precision/recall/F1, IoU)
│   │   └── split_metrics.py           # Backend-independent mAP on a dataset split
│   ├── inference/predictor.py       # Unified image/batch/video/webcam inference
│   └── preprocessing/
│       ├── perspective.py           # Perspective rectification (box rows, shelf sides)
│       └── stitch.py                # Video / photo stitching
├── configs/
│   ├── training_config.yaml           # Training hyperparameters
│   ├── dataset_config.yaml            # Dataset paths + class names
│   ├── bbox_crop_config.yaml           # Class filter for bbox_cropper.py
│   └── convert_class_id_map.yaml       # Class ID remap table
├── utils/helpers.py                 # Seeding, device detection, YAML I/O, GPU info
├── datasets/                        # Local dataset copies (populated by you)
├── logs/                            # Training logs
├── runs/                            # Training/inference outputs
├── train.py                         # Training entry point
├── infer.py                         # Inference entry point (image/video/webcam/GCS)
├── evaluate.py                      # Evaluation entry point
├── stitch.py                        # Video / photos -> wide still image (run before preprocess.py)
├── preprocess.py                    # Image preprocessor (perspective rectification, crop)
├── prepare_dataset.py               # Label Studio (GCS) → versioned GCS dataset
├── convert_xywhr_to_xywh.py         # CLI wrapper for rotated → axis-aligned boxes
├── convert_class_id_to_zero.py      # CLI wrapper to collapse all classes to id 0
├── convert_class_id_by_map.py       # CLI wrapper for custom class ID remapping
├── embed_class_mapping_in_model.py  # Bake a class ID remap into a model's names
├── validate_config.py               # Sanity-check training/dataset config YAMLs
├── examples.py                      # Runnable usage examples for the Python API
├── YOLO_Training_Notebook.ipynb     # Local/interactive training walkthrough
├── VertexAI_Training_Notebook.ipynb # Submit training as a Vertex AI custom job
├── requirements.txt                 # Runtime dependencies
└── requirements-dev.txt             # Dev tooling (pytest, black, flake8, isort)
```

## Installation

```bash
git clone <repository-url>
cd vizon_training

python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

For development (tests/formatting/linting):
```bash
pip install -r requirements-dev.txt
```

Verify the install:
```bash
python -c "from src.models import build_model; print('Installation successful!')"
python -c "import rfdetr"   # RF-DETR backend (installed via rfdetr[train] in requirements.txt)
```

> RF-DETR needs `numpy>=1.26` and torch ≥ 2.2, and its first use downloads pretrained weights
> to `~/.roboflow/models/` (internet access required).

Most GCS-backed commands below (`prepare_dataset.py`, `preprocessing.py`, `infer.py --mode
gcs_images`) need GCP credentials with access to the relevant buckets:
```bash
gcloud auth application-default login
```

## Workflow

### 1. Prepare a Dataset

**Target layout.** Whatever path you take below, you should end up with the standard YOLO
layout under `datasets/<name>/`:
```
datasets/my_dataset/
├── images/{train,val,test}/
└── labels/{train,val,test}/       # one .txt per image, same basename
```
Each label line is `<class_id> <x_center> <y_center> <width> <height>`, all coordinates
normalized to `[0, 1]`.

**Option A — From Label Studio annotations in GCS (`prepare_dataset.py`).**
Turns a Label Studio export into a versioned train/val/(test) dataset folder in GCS:

1. Lists `.txt` label files under
   `gs://dev-vveesee-ml-training/vizon-training/manual_annotations/detector/<gcs-annotation-folder>/labels/`
   (currently uploaded there manually).
2. Shuffles them with `--random-seed` and splits by `--train-ratio` / `--val-ratio` /
   `--test-ratio`. If `--test-ratio` is `0` (default), every non-train file goes to validation.
3. Matches each label to its source image (same basename, `.jpg`) under
   `gs://prd-vveesee-edge-submit/images/`.
4. Copies each image/label pair into
   `gs://dev-vveesee-ml-training/vizon-training/datasets/<gcs-destination-folder>/{images,labels}/{train,val,test}/`.
5. Writes `det_data_map.json` (destination image blob → destination label blob) back into the
   annotation folder, for downstream classification labelling.

```bash
python prepare_dataset.py \
    --gcs-annotation-folder rtd_batch1 \
    --gcs-destination-folder rtd \
    --train-ratio 0.7 \
    --val-ratio 0.3 \
    --test-ratio 0.0 \
    --random-seed 42
```

| Argument | Required | Default | Description |
|---|---|---|---|
| `--gcs-annotation-folder` | Yes | — | Subfolder under `manual_annotations/detector/` for this Label Studio export (must contain a `labels/` subfolder of `.txt` files); unique per Label Studio task. |
| `--gcs-destination-folder` | Yes | — | Subfolder under `datasets/` to create/populate, e.g. `rtd`. |
| `--train-ratio` | No | `0.7` | Fraction of files used for training. |
| `--val-ratio` | No | `0.3` | Fraction used for validation (ignored when `--test-ratio` is `0`; the remainder after train all goes to val). |
| `--test-ratio` | No | `0.0` | Fraction held out for testing. |
| `--random-seed` | No | `42` | Shuffle seed, for reproducible splits. |

Then pull the result down locally:
```bash
gsutil -m rsync -r gs://dev-vveesee-ml-training/vizon-training/datasets/<gcs-destination-folder> datasets/<name>
```

**Option B — Split an existing GCS labels folder (`src/data/preprocessing.py`).**
Use this when you already have a flat folder of labels in GCS (not a Label Studio export) and
just need a train/val/test split. It fetches each label's matching image (by file stem) from a
GCS images folder — default `gs://prd-vveesee-edge-submit/images` — and converts labels from
`class_id x y w h r` → `class_id x y w h` during the copy. Everything is read/written directly
to/from GCS; nothing is downloaded locally.

```bash
# Train/val split (80/20)
python src/data/preprocessing.py \
    --source-dir gs://dev-vveesee-ml-training/vizon-training/datasets/rtd_v7/labels \
    --output-dir gs://dev-vveesee-ml-training/vizon-training/datasets/rtd_v7_split
```

```bash
# Train/val/test split (70/20/10), custom images source and seed
python src/data/preprocessing.py \
    --source-dir gs://dev-vveesee-ml-training/vizon-training/datasets/rtd_v7/labels \
    --output-dir gs://dev-vveesee-ml-training/vizon-training/datasets/rtd_v7_split \
    --images-base-path gs://dev-vveesee-edge-submit/images \
    --train-pct 0.7 \
    --val-pct 0.2 \
    --test-pct 0.1 \
    --seed 123
```

Output (under `--output-dir`): `images/{train,val,test}/` and `labels/{train,val,test}/`
(`test/` only when `--test-pct > 0`).

Python API:
```python
from src.data.preprocessing import split_dataset

split_dataset(
    source_dir="gs://dev-vveesee-ml-training/vizon-training/manual_annotations/detector/my_task/labels",
    output_dir="gs://dev-vveesee-ml-training/vizon-training/datasets/my_dataset",
    train_pct=0.7,
    val_pct=0.2,
    test_pct=0.1,
)
```

**Option C — Bring your own local dataset.** Just arrange it into the target layout above.

### 2. Convert / Fix Labels

`src/data/label_converter.py` and its CLI wrappers handle common label-format fixups.

**Rotated boxes → YOLO OBB** (`class_id x y w h r` → OBB corner format):
```bash
python -c "from src.data.label_converter import convert_folder_xywhr_to_obb; \
convert_folder_xywhr_to_obb('input_labels', 'output_labels')"
```
```python
from src.data.label_converter import convert_folder_xywhr_to_obb, convert_file_xywhr_to_obb

convert_folder_xywhr_to_obb(
    input_dir='datasets/labels_rotated',
    output_dir='datasets/labels_obb',
    angle_unit='radians',  # or 'degrees'
    precision=6,
)
convert_file_xywhr_to_obb('labels/image1.txt', 'labels_obb/image1.txt')
```

**Rotated boxes → standard YOLO xywh** (`class_id x y w h r` → `class_id x y w h`):
```bash
python convert_xywhr_to_xywh.py --input-dir datasets/labels_rotated --output-dir datasets/labels
python convert_xywhr_to_xywh.py --input-dir in_labels --output-dir out_labels --precision 6
```
```python
from src.data.label_converter import convert_folder_xywhr_to_xywh, convert_file_xywhr_to_xywh

convert_folder_xywhr_to_xywh(input_dir='datasets/labels_rotated', output_dir='datasets/labels', precision=6)
convert_file_xywhr_to_xywh('labels/image1.txt', 'labels/image1_converted.txt', precision=6)
```

**Collapse all class IDs to 0** (single-class training):
```bash
python convert_class_id_to_zero.py --input-dir datasets/labels
python convert_class_id_to_zero.py --input-dir in_labels --output-dir out_labels
```
```python
from src.data.label_converter import convert_folder_class_id_to_zero, convert_file_class_id_to_zero

convert_folder_class_id_to_zero(input_dir='datasets/labels')                       # in-place
convert_folder_class_id_to_zero(input_dir='datasets/labels', output_dir='datasets/labels_zero_class')
convert_file_class_id_to_zero('labels/image1.txt', 'labels/image1_zero_class.txt')
```

**Remap class IDs by a custom table** (e.g. consolidating classes):

`configs/convert_class_id_map.yaml`:
```yaml
class_id_map:
  95: 0    # original class ID → new class ID
  101: 1
```
```bash
python convert_class_id_by_map.py --input-dir datasets/labels --map-config configs/convert_class_id_map.yaml
python convert_class_id_by_map.py --input-dir in_labels --output-dir out_labels --map-config configs/convert_class_id_map.yaml
```
```python
from pathlib import Path
from src.data.label_converter import convert_folder_class_id_by_map, load_class_id_map_from_yaml

class_id_map = load_class_id_map_from_yaml(Path('configs/convert_class_id_map.yaml'))
convert_folder_class_id_by_map(input_dir='datasets/labels', class_id_map=class_id_map)                          # in-place
convert_folder_class_id_by_map(input_dir='datasets/labels', class_id_map=class_id_map, output_dir='datasets/labels_remapped')

# or inline
convert_folder_class_id_by_map(input_dir='datasets/labels', class_id_map={95: 0, 101: 1, 102: 1})
```

### 3. Crop Bounding Boxes (optional)

Extract individual bounding boxes into per-class folders — useful for building a classifier
dataset or focused crops (`src/data/bbox_cropper.py`).

Optionally filter classes via `src/data/bbox_crop_config.yaml`:
```yaml
include_classes:
  - BOTTLE
  - JUG
  - REFILL POUCH
  - SACHET
```
(omit the config to include all classes)

```bash
python src/data/bbox_cropper.py \
    --images-dir datasets/homecare/images/train \
    --labels-dir datasets/homecare/labels/train \
    --output-dir ./crops_homecare \
    --classes-file datasets/homecare/classes.txt \
    --config-file src/data/bbox_crop_config.yaml \
    --padding 10
```
```python
from src.data.bbox_cropper import BBoxCropper

cropper = BBoxCropper(
    images_dir='datasets/homecare/images/train',
    labels_dir='datasets/homecare/labels/train',
    output_dir='./crops_homecare',
    classes_file='datasets/homecare/classes.txt',
    config_file='src/data/bbox_crop_config.yaml',  # optional
    padding=10,                                     # optional
)
stats = cropper.process_folder()
cropper.print_summary()
```
Output: `crops_homecare/<CLASS_NAME>/{image}_{n}.jpg`.

### 4. Configure the Dataset

Edit `configs/dataset_config.yaml`:
```yaml
path: ./datasets/rtd        # dataset root
train: ./images/train
val: ./images/val
test: ./images/test
nc: 2                       # number of classes
names:
  0: RTD
  1: Shelf
```

Sanity-check both config files before training:
```bash
python validate_config.py
```

### 5. Train

```bash
python train.py \
    --data configs/dataset_config.yaml \
    --config configs/training_config.yaml \
    --epochs 100 \
    --batch-size 32 \
    --imgsz 640 \
    --validate
```

Values not passed on the CLI fall back to `configs/training_config.yaml`, then to the script's
built-in defaults. The backend is chosen by `model_type` in that config (`yolo` or `rfdetr`).
For RF-DETR, set e.g. `model_type: 'rfdetr'` and `model_size: 'base'`; the YOLO-only flags below
(`--lr0`, `--lrf`, `--momentum`, `--weight-decay`, `--warmup-epochs`) and `--imgsz` are ignored,
and backend-specific arguments go in the config's `train_args:` block (e.g. `lr`,
`grad_accum_steps`). RF-DETR trains from your existing YOLO-format dataset: a `train/valid/test`
view is generated automatically next to the run folder (`runs/<save-dir>_dataset/`). Checkpoints are written to
`runs/<save-dir>/` (use `checkpoint_best_total.pth` for inference). Key flags:

| Argument | Default | Description |
|---|---|---|
| `--data` | `configs/dataset_config.yaml` | Dataset YAML |
| `--config` | `configs/training_config.yaml` | Training hyperparameter YAML (provides defaults for the flags below) |
| `--model-type` / `--model-size` | from config | Override `model_type` / `model_size` in the training config (e.g. `--model-type rfdetr --model-size base`) |
| `--epochs` | `100` | Training epochs |
| `--batch-size` | `32` | Batch size |
| `--imgsz` | `640` | Input image size |
| `--patience` | `20` | Early stopping patience |
| `--lr0` / `--lrf` | `0.01` / `0.01` | Initial LR / final LR ratio |
| `--momentum` | `0.937` | SGD momentum |
| `--weight-decay` | `0.0005` | L2 regularization |
| `--warmup-epochs` | `3` | LR warmup epochs |
| `--save-dir` | `runs/train` | Where results are written |
| `--log-dir` | `logs` | Where logs are written |
| `--no-cuda` | off | Force CPU |
| `--seed` | `42` | Random seed |
| `--validate` | off | Run validation after training |

### 6. Run Inference

`infer.py` supports four modes via `--mode`. Common flags across all modes: `--weights`
(required), `--conf` (default `0.5`), `--iou` (default `0.45`), `--device {cpu,cuda}` (default
`cpu`), `--class-id-map` (optional, see [below](#7-remap-class-ids-at-inference-time)),
`--model-type {yolo,rfdetr}` (default `yolo`) and `--model-size` (default: backend default, `m` for
yolo / `base` for rfdetr; must match the size the weights were trained with). RF-DETR is NMS-free,
so `--iou` has no effect with it.

**Single image** (`--mode image`). Add `--output out/` (folder) or `--output out/result.jpg` to also save
the image with boxes drawn:
```bash
python infer.py --weights runs/detect/train/weights/best.pt --source path/to/image.jpg --mode image
```

**Folder of images** (`--mode image` with a directory as `--source`; searched recursively, `--batch-size`
images per model call, default 8). All detections go into one JSON: `--results-json`, else
`<output>/results.json` when `--output` is given, else `<folder>_results.json` next to the folder.
With `--output out/`, an annotated copy of every image is saved there (sub-folders preserved):
```bash
python infer.py --weights runs/detect/train/weights/best.pt --source path/to/image_folder/ --mode image
python infer.py --model-type rfdetr --model-size base --weights models/rfdetr_base.pth \
    --source path/to/image_folder/ --mode image --output out/ --batch-size 8
```

**Video** (`--mode video`):
```bash
python infer.py \
    --weights runs/detect/train/weights/best.pt \
    --source path/to/video.mp4 \
    --mode video \
    --output output.mp4 \
    --fps 30
```

**Webcam** (`--mode webcam`):
```bash
python infer.py \
    --weights runs/detect/train/weights/best.pt \
    --mode webcam \
    --duration 30 \
    --output webcam_output.mp4
```

**A list of images in GCS** (`--mode gcs_images`) — batch inference straight from GCS, no
manual download step:

Create a text file listing one `gs://` image URI per line, then run:
```bash
python infer.py \
    --weights runs/detect/train/weights/best.pt \
    --source images.txt \
    --mode gcs_images \
    --output results.json
```
Each image is downloaded into a shared local cache (`--images-cache-dir`, default
`/home/enitze/projects/repos/vizon/shared_image_folder` — outside this repo, so it's shared
across runs and reused by `vizon_classifier_embedding`'s inference pipelines too) before being
predicted on; an image already present there (matched by the GCS blob's own filename) is reused
instead of re-downloaded. Detections are written to the output JSON keyed by `gcs_uri`. If
`--output` is omitted, results are written to `<images-file-stem>_results.json` next to
`images.txt`. Uses your default GCS credentials (`gcloud auth application-default login`).

### 7. Remap Class IDs at Inference Time

Two ways to apply a class ID remap (same `class_id_map` YAML format as
[label conversion](#2-convert--fix-labels)) without retraining:

**At inference time**, pass the map straight to `infer.py`:
```bash
python infer.py \
    --weights runs/detect/train/weights/best.pt \
    --source path/to/image.jpg \
    --mode image \
    --class-id-map configs/convert_class_id_map.yaml
```

**Baked into the model** (so every future load reflects the new IDs), via
`embed_class_mapping_in_model.py`:
```bash
python embed_class_mapping_in_model.py \
    --model models/best.pt \
    --mapping configs/inference_class_id_map.yaml \
    --output models/best_mapped.pt
```

### 8. Evaluate

```bash
python evaluate.py \
    --weights runs/detect/train/weights/best.pt \
    --test-dir datasets/my_dataset/images/test \
    --output eval_results.json \
    --conf 0.5 \
    --iou 0.5 \
    --device cuda
```

Add `--model-type rfdetr --model-size base` (and an RF-DETR checkpoint for `--weights`) to evaluate
an RF-DETR model. For mAP against ground-truth labels, use `model.val(data_yaml)` (see the API
reference): YOLO returns Ultralytics' metrics; RF-DETR returns `map`, `map50`, `map75`,
`per_class_map` and `num_images` for the `val` split (pass `split='test'` for the test split).

### Lensabonus scheduled inference (Cloud SQL -> local detection)

`lensabonus_inference_pipeline.py` mirrors the pipeline of the same name in `vizon_classifier_embedding`,
but runs this repo's detector (YOLO or RF-DETR) instead of a crop classifier:

1. Queries Cloud SQL (`transactions.user_activity`, `capture_annotate` rows) for a time window.
2. Downloads each image into the shared folder `/home/enitze/projects/repos/vizon/shared_image_folder`
   (same folder and `<row id>_<upload ts>_<filename>` naming as the embedding repo, so already-downloaded
   images are reused and never fetched twice).
3. Runs the detector locally, in batches.
4. Writes `infer/scheduled_runs/<timestamp>/{results.json,labels/*.txt,categories.json}`.
5. Writes lensabonus-format annotation files (`{"model_name", "model_predictions": [{"class_id", "box": {left,top,right,bottom}}]}`,
   normalized, at most 4 decimal digits) to `infer/annotations/<model name>/`. These are the hand-off to `vizon_classifier_embedding`:
   run its `lensabonus_inference_pipeline.py` with `--annotations-cache-dir <that folder>` and it crops and
   classifies *this detector's* boxes instead of lensabonus's (it finds them by the same `<row id>_*_<name>`
   glob, so no GCS annotation download happens). `--classes` there then refers to this model's class ids.
6. Only with `--write-class-map` (opt-in): upserts this model's class map into `labeling.class_maps`
   (`map_id` = `--model-name`, `{"0": "name", ...}`). This is the only thing written to the database; no
   `labeling.labels` rows are written here. The embedding pipeline writes the raw-detection `labels` rows
   (with `class_map` = the annotations' `model_name`) but never writes a class map for that name, it only
   reads it, so without this step those rows point at a map that doesn't exist and
   `box_labels.object_class_map_class` stays `NULL`. Run it once per model (it is an idempotent upsert).
   The Objects tab shows only the latest `objects` row per image (by `created_at`), so a newer AI row hides
   an older human-labeled one.

```bash
# last 24 h on dev (default window)
python lensabonus_inference_pipeline.py --weights models/rfdetr/rtd_rfdetr_v1/checkpoint_best_total.pth \
    --model-type rfdetr --model-size base --model-name rtd_rfdetr_v1
```

```bash
# explicit window and confidence threshold
python lensabonus_inference_pipeline.py --weights models/registered/rfdetr/trf_rtd_v1/rfdetr-base-DETECTION-20261029-092746_model_runs_checkpoint_best_total.pth \
    --model-type rfdetr --model-size base --model-name trf_rtd_v1 \
    --start-ts 2026-06-17T00:00:00 --end-ts 2026-07-31T00:00:00 --conf 0.4
```

```bash
# once per trained model: write its class map to labeling.class_maps (production)
python lensabonus_inference_pipeline.py --weights models/rfdetr/rtd_rfdetr_v1/checkpoint_best_total.pth \
    --model-type rfdetr --model-size base --model-name rtd_rfdetr_v1 --write-class-map --env prd
```

For a YOLO model use `--model-type yolo` (the default) with its `best.pt`, e.g.
`--weights models/yolo/rtd_yolo_v3/best.pt --model-name rtd_yolo_v3`; `--model-size` then defaults to `m`.

**Arguments**

| Argument | Required | Default | Description |
|---|---|---|---|
| `--weights` | yes | - | Path to the trained detector weights (`best.pt` for YOLO, `checkpoint_best_total.pth` for RF-DETR). |
| `--model-type` | no | `yolo` | Detector backend: `yolo` or `rfdetr` (any registered backend). |
| `--model-size` | no | backend default (`m` / `base`) | Model size; must match what the weights were trained with. |
| `--model-name` | no | `<model-type>_<weights file name>` | Stable identifier for this trained model, e.g. `rtd_yolo_v3`. Becomes the annotation files' `model_name` (-> `agent`/`class_map` in `labeling.labels` via the embedding pipeline), the default annotations folder name and the `--write-class-map` map id. The default is **not unique across trainings** (every YOLO run saves `best.pt`), so set it explicitly; a warning is printed when it is omitted. |
| `--device` | no | auto | `cpu` or `cuda`. |
| `--class-id-map` | no | none | YAML class-id mapping applied to the predictions. |
| `--conf` | no | `0.5` | Confidence threshold. |
| `--iou` | no | `0.45` | NMS IoU threshold. YOLO only; RF-DETR is NMS-free and ignores it. |
| `--batch-size` | no | `8` | Images per model call. |
| `--start-ts` | no | 24 h before `--end-ts` | ISO datetime, inclusive lower bound on the row's `created_at`. |
| `--end-ts` | no | now | ISO datetime, inclusive upper bound. |
| `--env` | no | `dev` | `dev` or `prd`: selects the Cloud SQL instance and the `<env>-vveesee-edge-submit` bucket. |
| `--images-cache-dir` | no | `/home/enitze/projects/repos/vizon/shared_image_folder` | Shared image folder (also used by `vizon_classifier_embedding`); images already there are not re-downloaded. |
| `--output-base-dir` | no | `infer/scheduled_runs` | Base folder; each run writes into a `<UTC timestamp>` sub-folder. |
| `--annotations-dir` | no | `infer/annotations/<model name>` | Where the lensabonus-format annotation files are written (point the embedding pipeline's `--annotations-cache-dir` here). |
| `--write-class-map` | no | off | Upsert the model's class map into `labeling.class_maps` (`map_id` = model name). Writes nothing else to the database. |

Needs ADC credentials (`gcloud auth application-default login`) with Secret Manager access to
`dev_pg_db` / `prd_pg_db` and read access to the `<env>-vveesee-edge-submit` bucket. RF-DETR runs with
Hugging Face offline mode forced on, so no Hugging Face calls are made.

### 9. Stitch Videos or Photos into Wide Images

`stitch.py` turns a horizontally moving video, or a few photos taken one after another along a
scene, into one wide still image. Run it **before** `preprocess.py`: its output folder can be
passed straight to `preprocess.py --source` for perspective correction and cropping. No model is
needed.

**Inputs** (`--source`, one or more paths):

| Input | Result |
|-------|--------|
| Video file | One image per video |
| Image files | All of them together are one photo set, stitched in the order given (`--name` sets the output name) |
| Folder (searched recursively) | Every video in it is stitched on its own, and every folder that directly holds two or more images is one photo set, in file-name order (`IMG_2` before `IMG_10`) |

Photos should be taken by stepping sideways along the shelf, facing it the same way each time,
with each photo overlapping the next by about a third or more. Two photos of the same bay from
very different angles can't be stitched: deep shelves and protruding products look too different
from each angle.

**Steps:**
1. **Keyframes** (videos only): the horizontal motion between frames is tracked. Each time the
   view has moved `1 - --overlap` of a frame width, the sharpest recent frame (least motion
   blur) becomes a keyframe. The first and last frames are always included.
2. **Stitching:** OpenCV's stitching pipeline (SIFT features, bundle adjustment, exposure
   compensation, graph-cut seams, multi-band blending), with each keyframe or photo matched
   only to its neighbours in order (`--match-range`). That keeps repeated products or shelf
   sections from being linked to the wrong place.
3. **Border crop:** the black border around the stitched area is cut off (`--no-crop` keeps it).

Videos with a rotation tag and photos with an EXIF orientation are turned upright first.

```bash
# A video (camera moving sideways), with the keyframes saved for checking
python stitch.py \
    --source path/to/shelf_video.mp4 \
    --output runs/stitched/ \
    --debug

# A few photos, left to right, in the order given
python stitch.py \
    --source shelf_1.jpg shelf_2.jpg shelf_3.jpg \
    --output runs/stitched/ \
    --name aisle_3

# A folder: each video, and each sub-folder of photos, becomes one image
#   visits/
#     aisle_1.mp4            -> runs/stitched/aisle_1.jpg
#     aisle_2/IMG_1..5.jpg   -> runs/stitched/aisle_2.jpg
python stitch.py \
    --source visits/ \
    --output runs/stitched/

# Camera turning on the spot over a wide angle, long 60 fps video
python stitch.py \
    --source path/to/video.mp4 \
    --output runs/stitched/ \
    --mode panorama \
    --frame-step 2

# Then rectify and crop the stitched images
python preprocess.py \
    --model-type rfdetr --model-size base \
    --weights models/rfdetr_base.pth \
    --source runs/stitched/ \
    --output runs/preprocess/ \
    --horizontal \
    --device cuda
```

Outputs in `--output`:

| Path | Contents |
|------|----------|
| `<name>.jpg` | Stitched image: `<video name>.jpg`, `<photo folder name>.jpg` (sub-folders of `--source` are preserved) or `--name` for listed photos |
| `debug/<name>_inputs.jpg` | With `--debug`: the keyframes / photos side by side, labelled (photos that couldn't be matched are marked `unused`) |
| `stitch.json` | Per output: inputs, keyframes, the ones used, mode, border crop, output size, and the reason when inputs were left out or stitching failed |

| Flag | Default | Meaning |
|------|---------|---------|
| `--name` | `<first image>_stitched` | Output name for image files listed in `--source` |
| `--mode` | `scans` | `scans`: camera moving sideways (or turning moderately); `panorama`: camera turning on the spot over a wide angle (straight lines curve slightly) |
| `--match-range` | `2` | Each keyframe / photo is matched with this many before and after it |
| `--confidence` | `0.4` | Min match confidence for two neighbouring inputs to be joined (pairs further apart need `1.0`, as a weak match there is usually wrong). Lower it (e.g. `0.3`) when inputs with little detail in common (plain walls, glass doors, reflections) are left out; raise it if parts are joined in wrong places |
| `--no-crop` | off | Keep the black border around the stitched area |
| `--jpeg-quality` | `95` | JPEG quality of the stitched image |
| `--overlap` | `0.6` | Videos: overlap between consecutive keyframes (fraction of the frame width) |
| `--max-frames` | `60` | Videos: at most this many keyframes are stitched (evenly thinned) |
| `--frame-step` | `1` | Videos: read only every n-th frame (faster on long / high-fps videos) |

If stitching fails with `alignment diverged` or `bundle adjustment failed`, try the other
`--mode`. When an input is left out, the message names it with its best match confidence: if
that is just below `--confidence`, lowering it usually works; if it is far below, the inputs
don't overlap enough.

### 10. Preprocess Images (perspective rectification, crop)

`preprocess.py` runs a chain of preprocessing steps (`--steps`, default both, in this order) on
an image or a folder of images (searched recursively; `debug/` sub-folders are skipped). For
videos or photo sets, run [`stitch.py`](#9-stitch-videos-or-photos-into-wide-images) first.

- **`perspective`**: detects boxes with the selected model and fits one regression line per row
  of boxes through each box's anchor point (`--anchor bottom|center|top`). Rows whose slope is an
  outlier against the others are dropped. The side the rows converge towards (the side
  farther from the camera) is stretched with a homography so the rows become parallel and
  horizontal.
  - With `--horizontal`, a second pass then makes the shelf unit's sides vertical. It uses only
    the shelf boxes (`--h-shelf-classes`) and first drops trimmed edge shelves (cut off by the
    left / right border and not reaching the image centre). It fits a line through the shelves'
    left ends and one through their right ends, and stretches the top or bottom edge they
    converge towards.
- **`crop`**: only for images the perspective step actually warped. The same model runs again on
  the corrected image, and the image is cropped (plus `--crop-buffer` px):
  - vertical pass: to the topmost box's top and the bottommost box's bottom, removing the black
    wedges above and below the shelf;
  - horizontal pass: to the shelf unit's two sides, which removes neighbouring fridges / bays.
    A side without a fitted line (e.g. cut off by the photo) falls back to the outermost box.
    Detections outside the crop are dropped.

```bash
# Single image, YOLO model, vertical correction + crop, with debug overlays
python preprocess.py \
    --weights runs/detect/train/weights/best.pt \
    --source path/to/image.jpg \
    --output runs/preprocess/ \
    --anchor bottom \
    --debug

# RF-DETR on GPU, vertical and horizontal correction (shelf sides), shelves excluded from the row fit
python preprocess.py \
    --model-type rfdetr --model-size base \
    --weights models/rfdetr_base.pth \
    --source path/to/image_folder/ \
    --output runs/preprocess/ \
    --anchor bottom \
    --exclude-classes Shelf \
    --horizontal \
    --device cuda \
    --debug
```

Outputs in `--output`:

| Path | Contents |
|------|----------|
| `<name>.<ext>` | Preprocessed image (sub-folders of `--source` are preserved) |
| `debug/<name>.<ext>` | With `--debug`: boxes, row lines (rejected rows thin grey) and the verdict, drawn on the input |
| `debug/<name>_crop.<ext>` | With `--debug`: detections on the corrected image and the crop limits |
| `debug/<name>_h.<ext>`, `debug/<name>_h_crop.<ext>` | Same for the horizontal pass (`--horizontal`); trimmed edge shelves are crossed out |
| `preprocess.json` | Per image: detections; `perspective` / `perspective_h` (convergence side, stretch ratio, vanishing point, homography, whether it was applied, rectified boxes, shelf-side positions); `crop` / `crop_h` (range, dropped boxes, detections after the crop); `output_detections` in output-image coordinates |

The CLI prints a summary such as
`Perspective (vertical): left: 12, not corrected: 5, right: 31 | cropped 43/48`.

Perspective flags (vertical pass):

| Flag | Default | Meaning |
|------|---------|---------|
| `--anchor` | `bottom` | Box point the row lines go through: `bottom`, `center` or `top` |
| `--min-boxes-per-row` | `3` | Rows with fewer boxes are ignored |
| `--row-tolerance` | `0.5` | How far (in median box heights) a box may be from a row's line and still join it |
| `--min-stretch-ratio` | `1.03` | Near/far ratio below which the image is left as is (`converges_to: none`) |
| `--max-stretch-ratio` | `3.0` | Ratio above which the fit is distrusted and the image is left as is |
| `--max-row-slope` | `0.5` | Rows steeper than this \|dy/dx\| are dropped as mis-grouped |
| `--slope-tolerance` | `0.03` | Rows whose slope deviates more than this from the other rows' trend are dropped as outliers |
| `--exclude-classes` | none | Class names left out of the row fit (e.g. `Shelf`) |

Horizontal pass flags:

| Flag | Default | Meaning |
|------|---------|---------|
| `--horizontal` | off | Also run the horizontal pass (after the vertical one) |
| `--h-shelf-classes` | `Shelf` | Class names whose boxes are shelves |
| `--h-edge-margin` | `0.01` | Shelf ends within this fraction of the width from the border count as cut off |
| `--h-tolerance` | `0.02` | Max distance of a shelf end from its side line (fraction of the width) |
| `--h-min-points` | `3` | Min shelf ends needed to fit a side |
| `--h-max-stretch-ratio` | `4.0` | Ratio above which the fit is distrusted and the image is left as is |

Crop flag:

| Flag | Default | Meaning |
|------|---------|---------|
| `--crop-buffer` | `10` | Pixels kept beyond the crop limits on each side |

Model flags (`--model-type`, `--model-size`, `--weights`, `--class-id-map`, `--conf`, `--iou`,
`--device`, `--batch-size`) work as in `infer.py`. An image is not corrected when too few rows (or shelf ends) qualify,
the stretch ratio is outside the min/max range, or the outer lines are too close together; the
reason is recorded in `preprocess.json`, and such images are not cropped.

## Notebooks

- **`YOLO_Training_Notebook.ipynb`** — interactive local walkthrough: install deps, prepare
  dataset, configure model, train, evaluate, save/load weights.
- **`VertexAI_Training_Notebook.ipynb`** — submit training as a managed Vertex AI custom job:
  configure, authenticate, (optionally) rebuild/push the training Docker image via Cloud Build,
  write the dataset config to GCS, submit and monitor the job, download results, then optionally
  run inference on the downloaded weights.

## Python API Reference

```python
from src.models import build_model, YOLOModel, RFDETRModel   # or BaseDetector for typing
from src.training import DetectorTrainer
from src.inference import DetectorPredictor
from src.evaluation import DetectorEvaluator
from src.data import YOLODataset, create_dataloader
from utils import setup_seed, get_device

setup_seed(42)
device = get_device()
```

**build_model / BaseDetector**

All backends implement `BaseDetector` (`train`, `val`, `predict`, `export`, `load_weights`).
`predict()` returns a list of `ImageDetections` (each with `detections` of `Detection`:
`class_id`, `class_name`, `confidence`, `x1..y2`) regardless of backend.
```python
model = build_model('rfdetr', 'base', device='cuda')    # or build_model('yolo', 'm')
model.load_weights('runs/train/checkpoint_best_total.pth')
metrics = model.val('configs/dataset_config.yaml')       # RF-DETR: {'map', 'map50', 'map75', ...}
```
To add a backend, subclass `BaseDetector` in `src/models/<name>_model.py` and add it to
`_BACKENDS` in `src/models/registry.py`.

**YOLOModel**
```python
model = YOLOModel(model_size='m', device='cuda')
model.load_weights('runs/detect/train/weights/best.pt')
results = model.train(data_yaml, epochs=100, batch_size=32)
predictions = model.predict(source='image.jpg', conf=0.5)
model.export(format='onnx')   # also supports 'torchscript', 'tflite'
```

**DetectorTrainer**
```python
trainer = DetectorTrainer(config_path='configs/training_config.yaml')
results = trainer.train(data_yaml='configs/dataset_config.yaml', epochs=100, batch_size=32)
trainer.validate(data_yaml)
trainer.log_metrics({'loss': 0.5})
```

**DetectorPredictor**
```python
predictor = DetectorPredictor(model, conf_threshold=0.5, iou_threshold=0.45)
results = predictor.predict_image('image.jpg')
results = predictor.predict_batch(['img1.jpg', 'img2.jpg'])
results = predictor.predict_video('video.mp4', output_path='out.mp4')
results = predictor.predict_webcam(duration=30)

for det in results['detections']:
    print(f"{det['class_name']}: {det['confidence']:.2f}")
```

**DetectorEvaluator**
```python
evaluator = DetectorEvaluator(model)
results = evaluator.evaluate('test_dir/', conf_threshold=0.5)
metrics = evaluator.compute_metrics(predictions, ground_truths)
```

**YOLODataset / DataLoader**
```python
dataset = YOLODataset(img_dir='images/train', label_dir='labels/train', img_size=640, augment=True)
loader = create_dataloader(img_dir='images/train', label_dir='labels/train', batch_size=32, shuffle=True)
```

See `examples.py` for runnable versions of the above.

## Configuration Reference

**`configs/training_config.yaml`** (defaults for `train.py`):
```yaml
model_type: 'yolo'     # yolo | rfdetr
model_size: 'm'        # yolo: n, s, m, l, x | rfdetr: nano, small, medium, base, large
# train_args:          # optional extra args passed to the backend's train()
#   lr: 0.0001           # rfdetr
#   grad_accum_steps: 4  # rfdetr
epochs: 100
batch_size: 8
imgsz: 640
patience: 20
single_cls: false      # treat all classes as one (yolo only)
lr0: 0.01              # lr0/lrf/momentum/weight_decay/warmup_epochs: yolo only
lrf: 0.01
# momentum, weight_decay, warmup_epochs, and augmentation params (hsv_*, degrees,
# translate, scale, flipud, fliplr, mosaic) are also supported — see the file for
# the full commented-out list.
device: null            # null = auto-detect
verbose: false
save_period: 10
data_dir: datasets
log_dir: logs
output_dir: train        # Ultralytics auto-increments: train, train1, train2, ... under runs/
```

**`configs/dataset_config.yaml`**:
```yaml
path: ./datasets/my_dataset
train: ./images/train
val: ./images/val
test: ./images/test
nc: 5
names:
  0: class1
  1: class2
```

## GPU & Performance

Check GPU availability:
```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"CUDA version: {torch.version.cuda}")
```
```python
from utils import print_gpu_info
print_gpu_info()
```

Force CPU: `python train.py --no-cuda`

**Batch size by GPU** (rule of thumb, `imgsz=640`): V100 16GB → 64, RTX 3090 → 128, T4 16GB → 32.
Rough VRAM floor by model size (batch_size=32, imgsz=640): nano 2GB+, small 4GB+, medium 6GB+,
large 12GB+.

**Image size vs. speed**: 640 = best accuracy/slowest, 416 = balanced, 320 = fastest/least
accurate.

**Model size vs. speed/accuracy**: nano (~3.2M params, fastest) < small (~11.2M) < medium
(~25.9M, balanced) < large (~52.9M) < xlarge (~71.4M, slowest/best).

## Troubleshooting

| Issue | Fix |
|---|---|
| `ModuleNotFoundError: No module named 'src'` | Run from the project root; if needed, `export PYTHONPATH="${PYTHONPATH}:$(pwd)"` |
| RF-DETR `ImportError` | `pip install "rfdetr[train]"` (the `[train]` extra is needed for training) |
| RF-DETR training OOM on a small GPU | Use `--batch-size 1` with `train_args: {grad_accum_steps: N}`; base fit in ~1.3 GB at batch 1 |
| Out of memory (OOM) during training | Reduce `--batch-size`, reduce `--imgsz`, or use a smaller `model_size` (n/s instead of l/x) |
| Training slow | Check `nvidia-smi` for GPU utilization, verify CUDA install, check dataloader `num_workers`, reduce augmentation |
| Poor accuracy | More/better-labeled training data, tune augmentation, tune hyperparameters in `training_config.yaml` |
| Dataset/label loading errors | Confirm image/label basenames match, labels are in YOLO format, coordinates are normalized to `[0, 1]`, and `dataset_config.yaml` paths are correct |
| GCS commands fail with auth errors | `gcloud auth application-default login`, and confirm your account has access to the source/destination buckets |

## Development

```bash
pip install -r requirements-dev.txt   # pytest, black, flake8, isort
```

## References

- [Ultralytics YOLOv8 Docs](https://docs.ultralytics.com/)
- [YOLOv8 GitHub](https://github.com/ultralytics/ultralytics)
- [RF-DETR GitHub](https://github.com/roboflow/rf-detr)
- [PyTorch Documentation](https://pytorch.org/docs/)
