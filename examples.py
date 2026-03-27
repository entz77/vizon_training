"""
Example usage of the YOLO training framework
"""

from src.models import YOLOModel
from src.training import YOLOTrainer
from src.inference import YOLOPredictor
from src.evaluation import YOLOEvaluator
from utils import setup_seed, get_device


def example_training():
    """Example: Training a YOLO model"""
    print("\n" + "=" * 50)
    print("Example: Training YOLO Model")
    print("=" * 50)
    
    # Setup
    setup_seed(42)
    device = get_device()
    
    # Initialize trainer
    trainer = YOLOTrainer(config_path='configs/training_config.yaml', task='detect')
    
    # Train (ensure dataset YAML exists)
    # results = trainer.train(
    #     data_yaml='configs/dataset_config.yaml',
    #     epochs=10,
    #     batch_size=16,
    #     imgsz=640
    # )
    # For oriented boxes, switch task to 'obb' and use OBB-formatted labels.
    
    print("Training example code ready (uncomment to run)")


def example_inference():
    """Example: Running inference"""
    print("\n" + "=" * 50)
    print("Example: YOLO Inference")
    print("=" * 50)
    
    # Load model
    model = YOLOModel(model_size='m', task='detect')
    
    # Initialize predictor
    predictor = YOLOPredictor(model, conf_threshold=0.5)
    
    # Predict on image
    # results = predictor.predict_image('path/to/image.jpg')
    # For OBB inference: YOLOModel(model_size='m', task='obb')
    # print(f"Detections: {len(results['detections'])}")
    
    print("Inference example code ready (uncomment to run)")


def example_evaluation():
    """Example: Model evaluation"""
    print("\n" + "=" * 50)
    print("Example: Model Evaluation")
    print("=" * 50)
    
    # Load model
    model = YOLOModel(model_size='m', task='detect')
    model.load_weights('path/to/weights.pt')
    
    # Evaluate
    evaluator = YOLOEvaluator(model)
    # results = evaluator.evaluate('path/to/test/images')
    # evaluator.save_results('eval_results.json')
    
    print("Evaluation example code ready (uncomment to run)")


def example_custom_data():
    """Example: Using custom dataset"""
    print("\n" + "=" * 50)
    print("Example: Custom Dataset")
    print("=" * 50)
    
    from src.data import create_dataloader
    
    # Create dataloader
    # train_loader = create_dataloader(
    #     img_dir='datasets/custom/images/train',
    #     label_dir='datasets/custom/labels/train',
    #     batch_size=32,
    #     shuffle=True
    # )
    
    print("Custom dataset example code ready (uncomment to run)")


if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("YOLO Training Framework Examples")
    print("=" * 60)
    
    example_training()
    example_inference()
    example_evaluation()
    example_custom_data()
    
    print("\n" + "=" * 60)
    print("Examples prepared. Uncomment code in example_*.py to use.")
    print("=" * 60)
