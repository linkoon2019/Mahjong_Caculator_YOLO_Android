# 基于 YOLOv11 的麻将牌识别与计分系统

本项目是一个结合了计算机视觉和移动端开发的综合系统，旨在通过手机摄像头实时检测麻将手牌，并根据麻将规则自动计算番数和点数。项目包含模型训练（Python/YOLO）和移动端应用（Android/Kotlin）两大部分。

**数据集致谢**: 本项目使用的原始数据集来自 Kaggle 开源项目 [Mahjong Hand Photos Taken with Mobile Camera](https://www.kaggle.com/datasets/shinz114514/mahjong-hand-photos-taken-with-mobile-camera/data)，感谢原作者的贡献。

## 🎯 项目概览

### 核心功能
*   **麻将牌检测**: 使用 YOLOv11 模型在复杂背景下识别麻将牌（支持万、筒、索、字牌及红宝牌）。
*   **自动计分**: Android 应用内置算番引擎，支持立直、自摸、宝牌等多种役种的计算。
*   **移动端部署**: 模型经过 TFLite 量化转换，可在 Android 设备上离线运行。

### 📁 项目结构

```
├── Mahjong-YOLO-main/           # 深度学习部分
│   ├── models/                  # 模型权重文件 (Nano/Small/Medium)
│   ├── notebooks/               # 数据处理与训练脚本 (Jupyter)
│   │   ├── data_labeling/       # 数据增强与标注处理
│   │   └── yolo.ipynb           # 模型训练主程序
│   ├── scripts/                 # 模型转换工具 (ONNX/CoreML)
│   └── trained_models_v2/       # 最终训练好的最佳模型
├── MahjongCalculator/           # Android 应用部分
│   ├── app/src/main/java/       # Kotlin 源代码
│   │   ├── YoloDetector.kt      # TFLite 推理封装
│   │   ├── MahjongCalculator.kt # 算番逻辑核心
│   │   └── MainActivity.kt      # UI 交互与相机逻辑
│   └── app/src/main/assets/     # TFLite 模型文件存放处
```

## 🚀 模型性能

本项目训练了多个不同规模的 YOLOv11 模型，以平衡速度与精度：

| 模型大小 | mAP@0.5 | mAP@0.5-95 | 适用场景 |
|:---:|:---:|:---:|:---|
| **Nano** | 0.880 | 0.676 | 极致速度，适合低端设备 |
| **Small** | **0.881** | **0.695** | 实时性与精度的最佳平衡 |
| **Medium** | 0.865 | 0.652 | 更高的召回率 |

## 🛠️ 快速开始

### 1. Python 环境 (模型训练与验证)

```bash
# 安装依赖
pip install ultralytics opencv-python matplotlib torch torchvision albumentations
```

### 2. Android 应用 (部署体验)

1.  使用 Android Studio 打开 `MahjongCalculator` 目录。
2.  连接真机并开启 USB 调试。
3.  点击 "Run" 即可安装应用。
    *   *注：应用需要相机权限以拍摄手牌。*

## 📊 数据集与增强

为了提高模型在真实场景下的鲁棒性，我们对原始 Kaggle 数据集进行了以下增强处理：
*   **几何变换**: 随机旋转 (±30°)、水平翻转、随机裁剪。
*   **环境模拟**: 亮度/对比度调整、高斯模糊、ISO 噪声模拟。
*   **遮挡模拟**: 随机区域挖空 (Cutout)，模拟反光或部分遮挡。

## 📄 许可证

本项目采用 MIT 许可证。
