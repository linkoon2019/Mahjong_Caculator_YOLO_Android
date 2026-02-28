#!/usr/bin/env python3
"""
Generate inference examples using trained YOLOv11 models
Demonstrates detection performance on mahjong tiles
"""
import os
import sys
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import random

def setup_directories():
    """Create directories for output"""
    inference_dir = Path("inference_examples")
    inference_dir.mkdir(exist_ok=True)
    return inference_dir

def get_test_images(test_dir, num_samples=5):
    """Get random test images for inference"""
    test_path = Path(test_dir)
    image_files = list(test_path.glob("*.png")) + list(test_path.glob("*.jpg"))

    if len(image_files) < num_samples:
        return image_files

    return random.sample(image_files, num_samples)

def visualize_predictions(image_path, results, model_name, save_path):
    """Visualize YOLO predictions on image"""
    # Load image
    image = cv2.imread(str(image_path))
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Create matplotlib figure
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.imshow(image_rgb)
    ax.set_title(f'{model_name} - Mahjong Tile Detection', fontsize=14, fontweight='bold')
    ax.axis('off')

    # Colors for different classes
    np.random.seed(42)  # For consistent colors
    colors = plt.cm.Set3(np.linspace(0, 1, 38))  # 38 classes

    # Draw predictions
    if len(results) > 0 and results[0].boxes is not None:
        boxes = results[0].boxes
        for i in range(len(boxes)):
            # Get box coordinates
            x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()
            conf = boxes.conf[i].cpu().numpy()
            cls = int(boxes.cls[i].cpu().numpy())

            # Skip low confidence detections
            if conf < 0.3:
                continue

            # Get class name
            class_names = ['1m', '1p', '1s', '1z', '2m', '2p', '2s', '2z',
                          '3m', '3p', '3s', '3z', '4m', '4p', '4s', '4z',
                          '5m', '5p', '5s', '5z', '6m', '6p', '6s', '6z',
                          '7m', '7p', '7s', '7z', '8m', '8p', '8s',
                          '9m', '9p', '9s', 'UNKNOWN', '0m', '0p', '0s']

            class_name = class_names[cls] if cls < len(class_names) else f'class_{cls}'

            # Draw bounding box
            width = x2 - x1
            height = y2 - y1
            rect = patches.Rectangle((x1, y1), width, height,
                                   linewidth=2, edgecolor=colors[cls],
                                   facecolor='none', alpha=0.8)
            ax.add_patch(rect)

            # Add label
            label = f'{class_name}\n{conf:.2f}'
            ax.text(x1, y1-5, label, fontsize=8, color=colors[cls],
                   fontweight='bold', bbox=dict(boxstyle="round,pad=0.3",
                   facecolor='white', alpha=0.7))

    # Add model info
    ax.text(0.02, 0.98, f'Model: {model_name}', transform=ax.transAxes,
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow', alpha=0.7))

    # Save figure
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    return len(boxes) if results[0].boxes is not None else 0

def generate_model_comparison(image_path, models, output_dir):
    """Generate comparison across all models for one image"""
    image_name = Path(image_path).stem

    # Create subplots for comparison
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.ravel()

    # Load image
    image = cv2.imread(str(image_path))
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    detection_counts = {}

    for idx, (model_name, model) in enumerate(models.items()):
        if idx >= 4:  # Only show 4 models max
            break

        ax = axes[idx]
        ax.imshow(image_rgb)
        ax.set_title(f'{model_name}', fontsize=12, fontweight='bold')
        ax.axis('off')

        # Run inference
        try:
            results = model(image_path, verbose=False)

            # Colors for different classes
            np.random.seed(42)
            colors = plt.cm.Set3(np.linspace(0, 1, 38))

            detection_count = 0

            if len(results) > 0 and results[0].boxes is not None:
                boxes = results[0].boxes
                for i in range(len(boxes)):
                    x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()
                    conf = boxes.conf[i].cpu().numpy()
                    cls = int(boxes.cls[i].cpu().numpy())

                    if conf < 0.3:
                        continue

                    detection_count += 1

                    # Draw bounding box
                    width = x2 - x1
                    height = y2 - y1
                    rect = patches.Rectangle((x1, y1), width, height,
                                           linewidth=1.5, edgecolor=colors[cls],
                                           facecolor='none', alpha=0.8)
                    ax.add_patch(rect)

            detection_counts[model_name] = detection_count

            # Add detection count
            ax.text(0.02, 0.98, f'Detections: {detection_count}',
                   transform=ax.transAxes, fontsize=10,
                   verticalalignment='top',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.7))

        except Exception as e:
            ax.text(0.5, 0.5, f'Error: {str(e)[:50]}...',
                   transform=ax.transAxes, ha='center', va='center',
                   fontsize=10, color='red')
            detection_counts[model_name] = 0

    # Hide unused subplots
    for idx in range(len(models), 4):
        axes[idx].axis('off')

    plt.suptitle(f'Model Comparison - {image_name}', fontsize=14, fontweight='bold')
    plt.tight_layout()

    # Save comparison
    comparison_path = output_dir / f"comparison_{image_name}.png"
    plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
    plt.close()

    return detection_counts

def main():
    """Main inference generation function"""
    print("🔍 Generating inference examples for trained YOLOv11 models...")

    # Setup
    output_dir = setup_directories()
    test_images_dir = "notebooks/data/processed/test/images"

    # Load models
    models = {}
    model_files = {
        'YOLOv11n': 'trained_models_v2/yolo11n_best.pt',
        'YOLOv11s': 'trained_models_v2/yolo11s_best.pt',
        'YOLOv11m': 'trained_models_v2/yolo11m_best.pt'
    }

    print("📦 Loading trained models...")
    for name, path in model_files.items():
        if os.path.exists(path):
            try:
                models[name] = YOLO(path)
                print(f"  ✅ {name}: {path}")
            except Exception as e:
                print(f"  ❌ Failed to load {name}: {e}")
        else:
            print(f"  ⚠️ Model not found: {path}")

    if not models:
        print("❌ No models available for inference!")
        return False

    # Get test images
    print(f"🖼️ Selecting test images from {test_images_dir}...")
    test_images = get_test_images(test_images_dir, num_samples=3)

    if not test_images:
        print("❌ No test images found!")
        return False

    print(f"  Found {len(test_images)} test images")

    # Generate individual model predictions
    print("🎯 Generating individual model predictions...")
    for model_name, model in models.items():
        print(f"\n📊 Processing with {model_name}...")

        for i, image_path in enumerate(test_images):
            try:
                print(f"  Processing {image_path.name}...")

                # Run inference
                results = model(str(image_path), verbose=False)

                # Save visualization
                output_name = f"{model_name.lower()}_{image_path.stem}.png"
                output_path = output_dir / output_name

                detection_count = visualize_predictions(
                    image_path, results, model_name, output_path
                )

                print(f"    ✅ Saved: {output_path} ({detection_count} detections)")

            except Exception as e:
                print(f"    ❌ Error processing {image_path.name}: {e}")

    # Generate model comparisons
    print("\n🔄 Generating model comparisons...")
    all_detection_counts = []

    for i, image_path in enumerate(test_images):
        try:
            print(f"  Comparing models on {image_path.name}...")
            detection_counts = generate_model_comparison(image_path, models, output_dir)
            all_detection_counts.append({
                'image': image_path.name,
                'counts': detection_counts
            })
            print(f"    ✅ Comparison saved")

        except Exception as e:
            print(f"    ❌ Error in comparison: {e}")

    # Generate summary
    print("\n📈 Generating summary report...")
    summary_path = output_dir / "inference_summary.txt"

    with open(summary_path, 'w') as f:
        f.write("YOLOv11 Mahjong Detection - Inference Summary\n")
        f.write("=" * 50 + "\n\n")

        f.write(f"Models tested: {list(models.keys())}\n")
        f.write(f"Test images: {len(test_images)}\n\n")

        f.write("Detection counts per image:\n")
        f.write("-" * 30 + "\n")

        for result in all_detection_counts:
            f.write(f"\nImage: {result['image']}\n")
            for model, count in result['counts'].items():
                f.write(f"  {model}: {count} detections\n")

        # Average detections
        if all_detection_counts:
            f.write("\nAverage detections per model:\n")
            f.write("-" * 30 + "\n")
            for model in models.keys():
                avg_detections = np.mean([
                    result['counts'].get(model, 0)
                    for result in all_detection_counts
                ])
                f.write(f"  {model}: {avg_detections:.1f}\n")

    print(f"✅ Summary saved: {summary_path}")

    # List all generated files
    generated_files = list(output_dir.glob("*.png")) + list(output_dir.glob("*.txt"))
    print(f"\n🎉 Generated {len(generated_files)} files in {output_dir}/:")
    for file in sorted(generated_files):
        print(f"  - {file.name}")

    return True

if __name__ == "__main__":
    success = main()
    if success:
        print("\n✨ Inference examples generated successfully!")
        print("📁 Check the 'inference_examples/' directory for results")
    else:
        print("\n❌ Failed to generate inference examples")
        sys.exit(1)