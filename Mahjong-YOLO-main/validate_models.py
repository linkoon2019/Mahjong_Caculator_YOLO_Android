#!/usr/bin/env python3
"""
Validate trained YOLOv11 models and extract performance metrics
"""
import os
from ultralytics import YOLO
import pandas as pd
from pathlib import Path

def validate_model(model_path, data_yaml):
    """Validate a single model and return metrics"""
    try:
        print(f"Validating {model_path}...")
        model = YOLO(model_path)

        # Run validation
        results = model.val(data=data_yaml, verbose=False)

        # Extract key metrics
        metrics = {
            'model': Path(model_path).stem,
            'precision': round(results.box.mp, 4),  # mean precision
            'recall': round(results.box.mr, 4),     # mean recall
            'mAP50': round(results.box.map50, 4),   # mAP at IoU=0.5
            'mAP50_95': round(results.box.map, 4),  # mAP at IoU=0.5:0.95
            'model_size_mb': round(os.path.getsize(model_path) / (1024*1024), 1)
        }

        print(f"  ✅ Precision: {metrics['precision']:.3f}")
        print(f"  ✅ Recall: {metrics['recall']:.3f}")
        print(f"  ✅ mAP50: {metrics['mAP50']:.3f}")
        print(f"  ✅ mAP50-95: {metrics['mAP50_95']:.3f}")

        return metrics

    except Exception as e:
        print(f"  ❌ Validation failed: {e}")
        return None

def main():
    """Main validation function"""
    print("🔍 Validating trained YOLOv11 models...")

    data_yaml = 'notebooks/data/processed/data.yaml'
    models_dir = 'trained_models_v2'

    # Model files to validate
    model_files = [
        f'{models_dir}/yolo11n_best.pt',
        f'{models_dir}/yolo11s_best.pt',
        f'{models_dir}/yolo11m_best.pt'
    ]

    results = []

    for model_path in model_files:
        if os.path.exists(model_path):
            metrics = validate_model(model_path, data_yaml)
            if metrics:
                results.append(metrics)
        else:
            print(f"⚠️ Model not found: {model_path}")

    if results:
        # Create results DataFrame
        df = pd.DataFrame(results)

        # Save to CSV
        df.to_csv('model_validation_results.csv', index=False)

        # Print summary table
        print("\n📊 Model Performance Summary:")
        print("=" * 70)
        print(f"{'Model':<12} {'Precision':<10} {'Recall':<8} {'mAP50':<8} {'mAP50-95':<10} {'Size(MB)':<10}")
        print("-" * 70)

        for _, row in df.iterrows():
            print(f"{row['model']:<12} {row['precision']:<10} {row['recall']:<8} {row['mAP50']:<8} {row['mAP50_95']:<10} {row['model_size_mb']:<10}")

        print("\nBest performing model (by mAP50):")
        best_model = df.loc[df['mAP50'].idxmax()]
        print(f"🏆 {best_model['model']}: mAP50={best_model['mAP50']:.3f}, mAP50-95={best_model['mAP50_95']:.3f}")

        return df
    else:
        print("❌ No models successfully validated")
        return None

if __name__ == "__main__":
    main()