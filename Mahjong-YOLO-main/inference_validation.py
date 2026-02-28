#!/usr/bin/env python3
"""
Comprehensive inference validation script
Compares model predictions with ground truth labels
"""
import os
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random
from collections import defaultdict

def get_class_names():
    """Get the 38 mahjong tile class names in correct order"""
    return ['1m', '1p', '1s', '1z', '2m', '2p', '2s', '2z',
            '3m', '3p', '3s', '3z', '4m', '4p', '4s', '4z',
            '5m', '5p', '5s', '5z', '6m', '6p', '6s', '6z',
            '7m', '7p', '7s', '7z', '8m', '8p', '8s',
            '9m', '9p', '9s', 'UNKNOWN', '0m', '0p', '0s']

def load_ground_truth_labels(label_file, img_width, img_height):
    """Load ground truth labels from YOLO format"""
    labels = []
    if os.path.exists(label_file):
        with open(label_file, 'r') as f:
            for line in f:
                if line.strip():
                    parts = line.strip().split()
                    class_id = int(float(parts[0]))
                    x_center = float(parts[1]) * img_width
                    y_center = float(parts[2]) * img_height
                    width = float(parts[3]) * img_width
                    height = float(parts[4]) * img_height

                    # Convert to x1, y1, x2, y2
                    x1 = x_center - width / 2
                    y1 = y_center - height / 2
                    x2 = x_center + width / 2
                    y2 = y_center + height / 2

                    labels.append({
                        'class_id': class_id,
                        'bbox': [x1, y1, x2, y2],
                        'confidence': 1.0  # Ground truth has 100% confidence
                    })
    return labels

def run_inference(model, image_path, conf_threshold=0.3):
    """Run inference on image"""
    results = model(str(image_path), verbose=False, conf=conf_threshold)

    predictions = []
    if len(results) > 0 and results[0].boxes is not None:
        boxes = results[0].boxes
        for i in range(len(boxes)):
            x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()
            conf = boxes.conf[i].cpu().numpy()
            cls = int(boxes.cls[i].cpu().numpy())

            predictions.append({
                'class_id': cls,
                'bbox': [x1, y1, x2, y2],
                'confidence': conf
            })

    return predictions

def calculate_iou(box1, box2):
    """Calculate IoU between two bounding boxes"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    if x2 <= x1 or y2 <= y1:
        return 0.0

    intersection = (x2 - x1) * (y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0.0

def match_predictions_to_labels(predictions, ground_truth, iou_threshold=0.5):
    """Match predictions to ground truth labels"""
    matched_pairs = []
    unmatched_preds = list(predictions)
    unmatched_labels = list(ground_truth)

    # For each ground truth, find best matching prediction
    for gt_idx, gt in enumerate(ground_truth):
        best_iou = 0
        best_pred_idx = -1

        for pred_idx, pred in enumerate(unmatched_preds):
            iou = calculate_iou(gt['bbox'], pred['bbox'])
            if iou > iou_threshold and iou > best_iou:
                best_iou = iou
                best_pred_idx = pred_idx

        if best_pred_idx >= 0:
            matched_pairs.append({
                'ground_truth': gt,
                'prediction': unmatched_preds[best_pred_idx],
                'iou': best_iou,
                'class_match': gt['class_id'] == unmatched_preds[best_pred_idx]['class_id']
            })
            unmatched_preds.pop(best_pred_idx)
            unmatched_labels.remove(gt)

    return matched_pairs, unmatched_preds, unmatched_labels

def visualize_comparison(image_path, predictions, ground_truth, class_names, output_path):
    """Create visualization comparing predictions vs ground truth"""

    # Load image
    image = cv2.imread(str(image_path))
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Create side-by-side comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

    # Ground Truth (Left)
    ax1.imshow(image_rgb)
    ax1.set_title('Ground Truth Labels', fontsize=14, fontweight='bold')
    ax1.axis('off')

    for gt in ground_truth:
        x1, y1, x2, y2 = gt['bbox']
        class_name = class_names[gt['class_id']]

        # Draw ground truth box in green
        rect = patches.Rectangle((x1, y1), x2-x1, y2-y1,
                               linewidth=2, edgecolor='green',
                               facecolor='none', alpha=0.8)
        ax1.add_patch(rect)

        # Add label
        ax1.text(x1, y1-5, f'GT: {class_name}', fontsize=9, color='green',
               fontweight='bold', bbox=dict(boxstyle="round,pad=0.3",
               facecolor='white', alpha=0.8))

    # Predictions (Right)
    ax2.imshow(image_rgb)
    ax2.set_title('Model Predictions', fontsize=14, fontweight='bold')
    ax2.axis('off')

    for pred in predictions:
        x1, y1, x2, y2 = pred['bbox']
        class_name = class_names[pred['class_id']] if pred['class_id'] < len(class_names) else f"class_{pred['class_id']}"
        conf = pred['confidence']

        # Draw prediction box in red
        rect = patches.Rectangle((x1, y1), x2-x1, y2-y1,
                               linewidth=2, edgecolor='red',
                               facecolor='none', alpha=0.8)
        ax2.add_patch(rect)

        # Add label
        ax2.text(x1, y1-5, f'Pred: {class_name}\n{conf:.3f}', fontsize=9, color='red',
               fontweight='bold', bbox=dict(boxstyle="round,pad=0.3",
               facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()

def main():
    """Main validation function"""
    print("🔍 Running comprehensive inference validation...")

    # Configuration
    model_path = 'trained_models_v2/yolo11m_best.pt'
    test_images_dir = 'notebooks/data/processed/test/images'
    test_labels_dir = 'notebooks/data/processed/test/labels'
    output_dir = Path('inference_validation')
    output_dir.mkdir(exist_ok=True)

    conf_threshold = 0.3
    iou_threshold = 0.5
    num_samples = 8  # Number of images to process

    # Load model
    print(f"📦 Loading model: {model_path}")
    try:
        model = YOLO(model_path)
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return False

    # Get class names
    class_names = get_class_names()

    # Get test images
    test_images = list(Path(test_images_dir).glob('*.png'))
    if len(test_images) < num_samples:
        print(f"⚠️ Only {len(test_images)} test images found")
        num_samples = len(test_images)

    # Randomly sample images
    selected_images = random.sample(test_images, num_samples)

    print(f"🎯 Processing {num_samples} test images...")

    # Statistics
    total_stats = {
        'total_images': 0,
        'total_gt_objects': 0,
        'total_predictions': 0,
        'correct_detections': 0,
        'class_correct': 0,
        'false_positives': 0,
        'false_negatives': 0,
        'class_stats': defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0})
    }

    # Process each image
    for i, image_path in enumerate(selected_images):
        print(f"\n📸 Processing {i+1}/{num_samples}: {image_path.name}")

        # Get corresponding label file
        label_file = Path(test_labels_dir) / f"{image_path.stem}.txt"

        # Load image to get dimensions
        image = cv2.imread(str(image_path))
        img_height, img_width = image.shape[:2]

        # Load ground truth
        ground_truth = load_ground_truth_labels(label_file, img_width, img_height)

        # Run inference
        predictions = run_inference(model, image_path, conf_threshold)

        # Match predictions to ground truth
        matches, unmatched_preds, unmatched_labels = match_predictions_to_labels(
            predictions, ground_truth, iou_threshold
        )

        # Calculate statistics for this image
        total_stats['total_images'] += 1
        total_stats['total_gt_objects'] += len(ground_truth)
        total_stats['total_predictions'] += len(predictions)
        total_stats['correct_detections'] += len(matches)
        total_stats['false_positives'] += len(unmatched_preds)
        total_stats['false_negatives'] += len(unmatched_labels)

        # Class-specific statistics
        for match in matches:
            if match['class_match']:
                total_stats['class_correct'] += 1
                class_name = class_names[match['ground_truth']['class_id']]
                total_stats['class_stats'][class_name]['tp'] += 1
            else:
                # Wrong class prediction
                gt_class = class_names[match['ground_truth']['class_id']]
                pred_class = class_names[match['prediction']['class_id']]
                total_stats['class_stats'][gt_class]['fn'] += 1
                total_stats['class_stats'][pred_class]['fp'] += 1

        # False positives
        for pred in unmatched_preds:
            pred_class = class_names[pred['class_id']]
            total_stats['class_stats'][pred_class]['fp'] += 1

        # False negatives
        for gt in unmatched_labels:
            gt_class = class_names[gt['class_id']]
            total_stats['class_stats'][gt_class]['fn'] += 1

        # Create visualization
        output_path = output_dir / f"validation_{i+1:02d}_{image_path.stem}.png"
        visualize_comparison(image_path, predictions, ground_truth, class_names, output_path)

        # Print results for this image
        print(f"  Ground Truth: {len(ground_truth)} objects")
        print(f"  Predictions: {len(predictions)} objects")
        print(f"  Matches: {len(matches)} (IoU > {iou_threshold})")
        print(f"  Class Correct: {sum(1 for m in matches if m['class_match'])}")
        print(f"  False Positives: {len(unmatched_preds)}")
        print(f"  False Negatives: {len(unmatched_labels)}")
        print(f"  ✅ Saved: {output_path}")

    # Calculate overall metrics
    print(f"\n📊 Overall Statistics:")
    print(f"=" * 50)
    print(f"Images processed: {total_stats['total_images']}")
    print(f"Total ground truth objects: {total_stats['total_gt_objects']}")
    print(f"Total predictions: {total_stats['total_predictions']}")
    print(f"Correct detections (IoU > {iou_threshold}): {total_stats['correct_detections']}")
    print(f"Class accuracy: {total_stats['class_correct']}/{total_stats['correct_detections']} = {total_stats['class_correct']/max(total_stats['correct_detections'], 1):.3f}")

    if total_stats['total_gt_objects'] > 0:
        recall = total_stats['correct_detections'] / total_stats['total_gt_objects']
        print(f"Recall: {recall:.3f}")

    if total_stats['total_predictions'] > 0:
        precision = total_stats['correct_detections'] / total_stats['total_predictions']
        print(f"Precision: {precision:.3f}")

    print(f"False Positives: {total_stats['false_positives']}")
    print(f"False Negatives: {total_stats['false_negatives']}")

    # Save detailed statistics
    stats_file = output_dir / "validation_statistics.txt"
    with open(stats_file, 'w') as f:
        f.write("Inference Validation Statistics\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"Model: {model_path}\n")
        f.write(f"Confidence threshold: {conf_threshold}\n")
        f.write(f"IoU threshold: {iou_threshold}\n")
        f.write(f"Images processed: {total_stats['total_images']}\n\n")

        f.write("Overall Metrics:\n")
        f.write(f"Total GT objects: {total_stats['total_gt_objects']}\n")
        f.write(f"Total predictions: {total_stats['total_predictions']}\n")
        f.write(f"Correct detections: {total_stats['correct_detections']}\n")
        f.write(f"Class accuracy: {total_stats['class_correct']}/{total_stats['correct_detections']} = {total_stats['class_correct']/max(total_stats['correct_detections'], 1):.3f}\n")

        if total_stats['total_gt_objects'] > 0:
            f.write(f"Recall: {total_stats['correct_detections'] / total_stats['total_gt_objects']:.3f}\n")
        if total_stats['total_predictions'] > 0:
            f.write(f"Precision: {total_stats['correct_detections'] / total_stats['total_predictions']:.3f}\n")

        f.write(f"\nFalse Positives: {total_stats['false_positives']}\n")
        f.write(f"False Negatives: {total_stats['false_negatives']}\n")

        f.write(f"\nPer-class statistics:\n")
        for class_name, stats in total_stats['class_stats'].items():
            tp, fp, fn = stats['tp'], stats['fp'], stats['fn']
            if tp + fp + fn > 0:
                precision = tp / (tp + fp) if tp + fp > 0 else 0
                recall = tp / (tp + fn) if tp + fn > 0 else 0
                f.write(f"{class_name}: TP={tp}, FP={fp}, FN={fn}, P={precision:.3f}, R={recall:.3f}\n")

    print(f"\n✅ Validation complete!")
    print(f"📁 Results saved in: {output_dir}/")
    print(f"📊 Statistics saved: {stats_file}")
    print(f"\nGenerated files:")
    for file in sorted(output_dir.glob("*.png")):
        print(f"  - {file.name}")

    return True

if __name__ == "__main__":
    random.seed(42)  # For reproducible sampling
    main()