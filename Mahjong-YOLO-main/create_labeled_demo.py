#!/usr/bin/env python3
"""
Create demo image with clear English labels and confidence scores
"""
import os
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def get_class_names():
    """Get the 38 mahjong tile class names"""
    return ['1m', '1p', '1s', '1z', '2m', '2p', '2s', '2z',
            '3m', '3p', '3s', '3z', '4m', '4p', '4s', '4z',
            '5m', '5p', '5s', '5z', '6m', '6p', '6s', '6z',
            '7m', '7p', '7s', '7z', '8m', '8p', '8s',
            '9m', '9p', '9s', 'UNKNOWN', '0m', '0p', '0s']

def get_tile_english_name(class_name):
    """Get English description of mahjong tile"""
    descriptions = {
        # Characters (Man/Wan)
        '1m': '1-Man', '2m': '2-Man', '3m': '3-Man', '4m': '4-Man', '5m': '5-Man',
        '6m': '6-Man', '7m': '7-Man', '8m': '8-Man', '9m': '9-Man', '0m': 'Red-5-Man',

        # Circles (Pin/Bing)
        '1p': '1-Pin', '2p': '2-Pin', '3p': '3-Pin', '4p': '4-Pin', '5p': '5-Pin',
        '6p': '6-Pin', '7p': '7-Pin', '8p': '8-Pin', '9p': '9-Pin', '0p': 'Red-5-Pin',

        # Bamboos (Sou/Tiao)
        '1s': '1-Sou', '2s': '2-Sou', '3s': '3-Sou', '4s': '4-Sou', '5s': '5-Sou',
        '6s': '6-Sou', '7s': '7-Sou', '8s': '8-Sou', '9s': '9-Sou', '0s': 'Red-5-Sou',

        # Honor tiles
        '1z': 'East', '2z': 'South', '3z': 'West', '4z': 'North',
        '5z': 'Red-Dragon', '6z': 'Green-Dragon', '7z': 'White-Dragon',

        'UNKNOWN': 'Unknown'
    }
    return descriptions.get(class_name, class_name)

def create_labeled_demo(image_path, model, output_path, conf_threshold=0.3):
    """Create demo with labeled detections"""

    print(f"Creating labeled demo from {image_path}...")

    # Load image
    image = cv2.imread(str(image_path))
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(16, 12))
    ax.imshow(image_rgb)
    ax.set_title('YOLOv11 Mahjong Tile Detection with Labels and Confidence Scores',
                 fontsize=16, fontweight='bold', pad=20)
    ax.axis('off')

    class_names = get_class_names()

    # Colors for different tile types
    type_colors = {
        'm': '#FF6B6B',      # Characters - Red
        'p': '#4ECDC4',      # Circles - Teal
        's': '#45B7D1',      # Bamboos - Blue
        'z': '#96CEB4',      # Honors - Green
        'UNKNOWN': '#FECA57', # Unknown - Yellow
        '0': '#FF8C42'       # Red fives - Orange
    }

    try:
        # Run inference
        results = model(str(image_path), verbose=False, conf=conf_threshold)

        detection_count = 0
        confidence_sum = 0
        detections_by_type = {}

        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes

            for i in range(len(boxes)):
                x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()
                conf = boxes.conf[i].cpu().numpy()
                cls = int(boxes.cls[i].cpu().numpy())

                if conf < conf_threshold:
                    continue

                detection_count += 1
                confidence_sum += conf

                # Get class info
                class_name = class_names[cls] if cls < len(class_names) else f'class_{cls}'
                tile_name = get_tile_english_name(class_name)

                # Determine color by tile type
                if class_name.startswith('0'):  # Red fives
                    color = type_colors['0']
                    tile_type = 'Red-5'
                elif class_name == 'UNKNOWN':
                    color = type_colors['UNKNOWN']
                    tile_type = 'Unknown'
                else:
                    tile_type = class_name[-1]  # Last character (m/p/s/z)
                    color = type_colors.get(tile_type, '#FECA57')

                # Count by type
                if tile_type not in detections_by_type:
                    detections_by_type[tile_type] = 0
                detections_by_type[tile_type] += 1

                # Draw bounding box
                width = x2 - x1
                height = y2 - y1
                rect = patches.Rectangle((x1, y1), width, height,
                                       linewidth=3, edgecolor=color,
                                       facecolor='none', alpha=0.9)
                ax.add_patch(rect)

                # Create label with name and confidence
                label_text = f'{tile_name}\n{conf:.3f}'

                # Position label ABOVE the detection box to avoid blocking the tile
                # Calculate label height and ensure it doesn't go off-screen
                label_height = 35  # Approximate height of label
                label_y = max(y1 - label_height - 5, label_height + 5)

                # If we're too close to top, place below the box instead
                if y1 < label_height + 10:
                    label_y = y2 + 5
                    valign = 'bottom'
                else:
                    label_y = y1 - 5
                    valign = 'top'

                # Add label with contrasting background - positioned to not block tile
                bbox_props = dict(boxstyle="round,pad=0.4",
                                facecolor=color, alpha=0.9,
                                edgecolor='white', linewidth=2)
                ax.text(x1, label_y, label_text, fontsize=10, color='white',
                       fontweight='bold', bbox=bbox_props,
                       verticalalignment=valign)

        # Add comprehensive statistics
        avg_conf = confidence_sum / detection_count if detection_count > 0 else 0

        stats_text = f'Model: YOLOv11m (Best Recall)\n'
        stats_text += f'Total Detections: {detection_count}\n'
        stats_text += f'Average Confidence: {avg_conf:.3f}\n'
        stats_text += f'Confidence Threshold: {conf_threshold}'

        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
               fontsize=12, verticalalignment='top', fontweight='bold',
               bbox=dict(boxstyle="round,pad=0.5", facecolor='white',
                        alpha=0.95, edgecolor='black'))

        # Add detection breakdown by type
        if detections_by_type:
            type_names = {
                'm': 'Characters', 'p': 'Circles', 's': 'Bamboos',
                'z': 'Honors', '0': 'Red-5s', 'Unknown': 'Unknown'
            }

            breakdown_text = 'Detections by Type:\n'
            for tile_type, count in sorted(detections_by_type.items()):
                type_name = type_names.get(tile_type, tile_type)
                breakdown_text += f'{type_name}: {count}\n'

            ax.text(0.98, 0.98, breakdown_text, transform=ax.transAxes,
                   fontsize=11, verticalalignment='top', horizontalalignment='right',
                   fontweight='bold',
                   bbox=dict(boxstyle="round,pad=0.5", facecolor='lightblue',
                            alpha=0.9, edgecolor='blue'))

    except Exception as e:
        ax.text(0.5, 0.5, f'Error: {str(e)}',
               transform=ax.transAxes, ha='center', va='center',
               fontsize=14, color='red', fontweight='bold')

    # Create legend for tile types
    legend_elements = []
    type_descriptions = {
        'm': 'Characters (万子)', 'p': 'Circles (筒子)', 's': 'Bamboos (条子)',
        'z': 'Honors (字牌)', '0': 'Red Fives (红五)', 'UNKNOWN': 'Unknown'
    }

    for tile_type, color in type_colors.items():
        if tile_type in type_descriptions:
            legend_elements.append(
                patches.Patch(color=color, label=type_descriptions[tile_type])
            )

    ax.legend(handles=legend_elements, loc='lower center', ncol=3,
             fontsize=11, bbox_to_anchor=(0.5, -0.1))

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)

    # Save high resolution image
    plt.savefig(output_path, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()

    print(f"✅ Labeled demo saved: {output_path}")
    return True

def main():
    """Create labeled demo image"""
    print("🎯 Creating labeled detection demo...")

    # Load YOLOv11m model (better recall for finding all tiles)
    model_path = 'trained_models_v2/yolo11m_best.pt'

    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        return False

    try:
        model = YOLO(model_path)
        print(f"✅ Loaded YOLOv11m model")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return False

    # Use same test image
    test_images_dir = "notebooks/data/processed/test/images"
    test_images = list(Path(test_images_dir).glob("*.png"))

    if not test_images:
        print("❌ No test images found!")
        return False

    # Find the specific image we've been using
    selected_image = None
    for img in test_images:
        if "58eec28b-000021" in str(img):
            selected_image = img
            break

    if not selected_image:
        selected_image = test_images[0]

    print(f"📸 Using test image: {selected_image.name}")

    # Create output
    output_dir = Path("inference_examples")
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "labeled_detection_demo.png"

    success = create_labeled_demo(selected_image, model, output_path)

    if success:
        print(f"\n🎉 Labeled demo created successfully!")
        print(f"📁 Output: {output_path}")
        print("\nFeatures:")
        print("  - Clear English tile names")
        print("  - Confidence scores for each detection")
        print("  - Color-coded by tile type")
        print("  - Detection statistics and breakdown")
        return True
    else:
        print("\n❌ Failed to create labeled demo")
        return False

if __name__ == "__main__":
    main()