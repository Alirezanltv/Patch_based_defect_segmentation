# Industrial Defect Detection Pipeline

This repository contains the implementation of a comprehensive three-stage defect detection pipeline for industrial applications. The pipeline includes defect detection, classification, and segmentation using custom neural network architectures.

## Overview

Our approach addresses the entire defect detection workflow through three hierarchical stages:

1. **Binary Classification**: Identifying whether industrial elements have defects (Inception40+Vgg16)
2. **Defect Type Classification**: Categorizing the specific type of defect (Dense27+Vgg16)
3. **Defect Segmentation**: Precisely localizing defects at the pixel level (UD52+DeepCNN)

This hierarchical approach significantly improves both accuracy and computational efficiency compared to single-stage approaches.

<p>
   <img width="1200" src="https://github.com/Alirezanltv/Patch_based_defect_segmentation/blob/main/results/Drawing1.jpg">
</p>

## Architecture

### Stage 1: Inception40+Vgg16 for Defect Classification

The first stage uses a hybrid architecture combining InceptionV3 and VGG16 networks to determine whether an image contains a defect, achieving **98.6%** accuracy.

### Stage 2: Dense27+Vgg16 for Defect Type Classification

For defective elements, the second stage uses a custom architecture combining DenseNet and VGG16 to classify the defect type among 8 different categories, achieving **97.4%** accuracy.

### Stage 3: UD52+DeepCNN for Defect Segmentation

The final stage uses a patch-based segmentation approach with a custom architecture combining UNet, DenseNet, and multi-scale CNN modules for precise defect localization.

<p>
   <img width="1200" src="https://github.com/Alirezanltv/Patch_based_defect_segmentation/blob/main/Last_Seg_Methodology.jpg">
</p>

## Patch-Based Approach

Our segmentation network uses a patch-based approach that divides input images into smaller patches (64×64) for processing. This technique offers several advantages:

1. **Data Augmentation**: Increases the effective size of the training dataset
2. **Memory Efficiency**: Allows processing of high-resolution images with limited GPU memory
3. **Improved Performance**: Helps the network focus on local defect features
4. **Better Generalization**: Reduces overfitting by creating more diverse training examples

<p>
   <img width="1200" src="https://github.com/Alirezanltv/Patch_based_defect_segmentation/blob/main/results/Drawing2.jpg">
</p>

## Multi-Scale Feature Extraction

Our segmentation architecture uses varying kernel sizes (3×3, 4×4, 5×5) in different layers to capture defects at multiple scales:

<p>
   <img width="1000" src="https://github.com/Alirezanltv/Patch_based_defect_segmentation/blob/main/results/different_kernel_1.PNG">
</p>

<p>
   <img width="1000" src="https://github.com/Alirezanltv/Patch_based_defect_segmentation/blob/main/results/different_kernel_2.jpg">
</p>

## Code Structure

- `inception40_vgg16_classifier.py`: Implementation of the first stage binary classifier
- `dense27_vgg16_classifier.py`: Implementation of the second stage defect type classifier
- `ud52_deepcnn_segmentation.py`: Implementation of the patch-based segmentation network
- `preprocessing.py`: CLAHE filter implementation and patch extraction utilities
- `utils.py`: Helper functions for data loading, evaluation metrics, and visualization

## Dataset Requirements

The code is designed to work with multiple industrial defect datasets:

- NEU Surface Defect Database
- Concrete Crack Images
- Micro Surface Defect
- Road-Defect
- DAGM-2007

Expected directory structure:
```
dataset/
├── train/
│   ├── defective/
│   │   └── [images]
│   └── non_defective/
│       └── [images]
├── test/
│   ├── defective/
│   │   └── [images]
│   └── non_defective/
│       └── [images]
```

For defect type classification, each defect category should have its own subdirectory.

## Performance

Our approach achieves state-of-the-art performance:

| Stage | Task | Accuracy |
|-------|------|----------|
| 1 | Defect Identification | 98.6% |
| 2 | Defect Type Classification | 97.4% |
| 3 | Defect Segmentation (DAGM 2007) | 81.78% MIoU |
| 3 | Defect Segmentation (Road Defect) | 80.01% MIoU |

## Usage

1. Clone the repository:
```bash
git clone https://github.com/Alirezanltv/Patch_based_defect_segmentation.git
cd Patch_based_defect_segmentation
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Train the models:
```bash
# Stage 1: Binary Classification
python Classification_part1.py --data_dir path/to/dataset

# Stage 2: Defect Type Classification
python Classification_part2.py --data_dir path/to/defect_types

# Stage 3: Segmentation
python ud52_deepcnn_segmentation.py --data_dir path/to/segmentation_data
```

4. Inference on new images:
```bash
python predict.py --input path/to/image.jpg
```

## Citation



## License

This project is licensed under the MIT License - see the LICENSE file for details.
