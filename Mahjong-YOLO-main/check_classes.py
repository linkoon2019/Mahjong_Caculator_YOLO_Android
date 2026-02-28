#!/usr/bin/env python3
"""
Check if the classes used in training match the data.yaml configuration
"""
import os
import yaml
from collections import Counter
from pathlib import Path

def load_data_yaml(yaml_path):
    """Load data.yaml configuration"""
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    return data

def check_label_files(labels_dir):
    """Check all label files and extract class IDs"""
    label_files = list(Path(labels_dir).glob("*.txt"))
    all_classes = []

    for label_file in label_files:
        with open(label_file, 'r') as f:
            for line in f:
                if line.strip():
                    class_id = int(float(line.strip().split()[0]))
                    all_classes.append(class_id)

    return all_classes

def main():
    """Main function to check class consistency"""
    print("🔍 Checking class configuration consistency...")

    # Load data.yaml
    data_yaml_path = 'notebooks/data/processed/data.yaml'
    data_config = load_data_yaml(data_yaml_path)

    print(f"\n📄 Data.yaml configuration:")
    print(f"  Number of classes (nc): {data_config['nc']}")
    print(f"  Class names: {data_config['names']}")
    print(f"  Expected class IDs: 0-{data_config['nc']-1}")

    # Check training labels
    train_labels_dir = 'notebooks/data/processed/train/labels'
    val_labels_dir = 'notebooks/data/processed/val/labels'
    test_labels_dir = 'notebooks/data/processed/test/labels'

    datasets = {
        'training': train_labels_dir,
        'validation': val_labels_dir,
        'test': test_labels_dir
    }

    for dataset_name, labels_dir in datasets.items():
        if os.path.exists(labels_dir):
            print(f"\n📊 {dataset_name.title()} dataset:")
            classes_found = check_label_files(labels_dir)

            if classes_found:
                class_counts = Counter(classes_found)
                min_class = min(classes_found)
                max_class = max(classes_found)
                unique_classes = len(set(classes_found))

                print(f"  Total annotations: {len(classes_found)}")
                print(f"  Unique classes found: {unique_classes}")
                print(f"  Class ID range: {min_class} - {max_class}")

                # Check if any class IDs are outside expected range
                expected_range = set(range(data_config['nc']))
                found_classes = set(classes_found)

                if found_classes <= expected_range:
                    print(f"  ✅ All class IDs are within expected range (0-{data_config['nc']-1})")
                else:
                    unexpected = found_classes - expected_range
                    print(f"  ❌ Unexpected class IDs found: {unexpected}")

                # Check if any expected classes are missing
                missing = expected_range - found_classes
                if missing:
                    print(f"  ⚠️ Missing class IDs: {sorted(missing)}")
                    # Map missing IDs to class names
                    missing_names = [data_config['names'][i] for i in sorted(missing) if i < len(data_config['names'])]
                    print(f"  ⚠️ Missing class names: {missing_names}")
                else:
                    print(f"  ✅ All expected classes are present")

                # Show top 10 most frequent classes
                print(f"  🔝 Top 10 most frequent classes:")
                for class_id, count in class_counts.most_common(10):
                    class_name = data_config['names'][class_id] if class_id < len(data_config['names']) else f'unknown_{class_id}'
                    print(f"    Class {class_id} ({class_name}): {count} instances")
            else:
                print(f"  ❌ No labels found")
        else:
            print(f"\n❌ {dataset_name.title()} labels directory not found: {labels_dir}")

    # Additional validation - check a trained model
    print(f"\n🤖 Checking trained model class configuration...")

    try:
        from ultralytics import YOLO
        model_path = 'trained_models_v2/yolo11m_best.pt'
        if os.path.exists(model_path):
            model = YOLO(model_path)
            model_names = model.names
            print(f"  Model class count: {len(model_names)}")
            print(f"  Model class names: {list(model_names.values())}")

            # Compare with data.yaml
            if list(model_names.values()) == data_config['names']:
                print(f"  ✅ Model classes match data.yaml exactly")
            else:
                print(f"  ❌ Model classes differ from data.yaml")
                print(f"  Expected: {data_config['names']}")
                print(f"  Model has: {list(model_names.values())}")
        else:
            print(f"  ⚠️ Model not found: {model_path}")
    except Exception as e:
        print(f"  ❌ Error loading model: {e}")

if __name__ == "__main__":
    main()