
# Perspective Occupancy Map (POM)

Predict semantic occupancy from monocular perspective images and project them to bird’s-eye view using a learned feature projection pipeline.

![Architecture](images/pom_architecture.svg)

## Overview

This repository implements **POMv2**, a deep learning framework for jointly predicting:
- **Semantic perspective occupancy maps** using DeepLabV3,
- **Projected top-view (BEV) segmentation** using learnable BEV fusion.

It uses PyTorch Lightning for modular training and evaluation, and integrates Weights & Biases (wandb) for experiment tracking.

## Key Features

- **Two-Branch Architecture:** Combines a segmentation branch (DeepLabV3) and a learned BEV encoder-decoder (PON branch).
- **Learned Perspective-to-BEV Projection:** Uses camera geometry to scatter semantic logits and fuse features for top-down decoding.
- **BEV-Aware Training:** Supervises both semantic occupancy maps and top-view segmentation.
- **Configurable & Reproducible:** YAML-based experiment configs and W&B integration.

## Model
POMv2 consists of a two-branch architecture designed to jointly predict semantic occupancy in both perspective view and top-down BEV:

- **Semantic POM Head:** A DeepLabV3 network predicts object footprints in perspective images. These footprints are more stable across frames than raw RGB, providing a consistent spatial signal that helps stabilize training and improves convergence.

- **PON Branch:** Extracts high-level geometric and semantic features directly from the perspective image using a CNN encoder. These features are projected into the BEV space and decoded via a UNet-style module.

- **BEV Fusion:** The model fuses projected semantic logits from the POM head and PON features into a unified BEV representation for final segmentation.

#### Why Predicting Perspective Occupancy Helps
- **Temporal Stability:** Object footprints in perspective views change slowly across adjacent frames, providing a strong inductive bias that improves generalization.

- **Improved Supervision:** Learning semantic POM maps provides intermediate supervision that grounds the BEV learning with spatial priors.

- **Cross-Dataset Transfer:** The POM head can be pretrained on large-scale segmentation datasets (e.g., Cityscapes), enabling the model to leverage diverse real-world data and generalize better to new environments.

## Architecture

The architecture contains:
- A **DeepLabV3** model for semantic POM prediction.
- A **PON_mod** encoder that extracts spatial features from the perspective view.
- A **BEV projection module** that maps semantic logits into a top-view grid using calibrated camera geometry.
- A **UNet-style decoder** that predicts top-down segmentation maps.

## Project Structure

```
.
├── configs/         # Experiment configs
├── datasets/        # Dataset definitions and transforms
├── models/          # Model components (POMv2, PON_mod, etc.)
├── utils/           # Helper functions and metrics
├── images/          # Architecture and output visualizations
├── train.py         # Training script
├── eval.py          # Evaluation script
├── requirements.txt # Package dependencies
└── README.md
```

## Installation

```bash
git clone https://github.com/yourusername/perspective-occupancy-map.git
cd perspective-occupancy-map
pip install -r requirements.txt
```

## Training

Configure your experiment in `configs/*.yaml`, then run:

```bash
python train.py --config configs/your_config.yaml
```

## Evaluation

To run evaluation on a trained checkpoint:

```bash
python eval.py --config configs/your_config.yaml --ckpt path/to/checkpoint.ckpt
```

## Results (TBA)
| Dataset            | Segmentation Objects | mIOU(%) | mAP(%)| Pretrained Model                                                                                                       | 
| :--------:           | :-----:     | :----:   | :----: | :----:                                                                                                                 |
| KITTI 3D Object     | Vehicle    |  -  | - | - |
| KITTI Odometry     | Road     |  -  | - | - |
| KITTI Raw          | Road     |  -  | - | - |
| Argoverse Tracking | Vehicle    |  -  | - | - |
| Argoverse Tracking | Road    |  -  | - | - |



## License

This project is licensed under the MIT License. See LICENSE for details.

## Citation

If you use this project in your research, please cite us.

```bibtex
@software{Singh_Perspective_Occupancy_Map_2022,
author = {Singh, Shantanu},
doi = {10.5281/zenodo.16370976},
month = aug,
title = {{Perspective Occupancy Map (POM)}},
url = {https://github.com/shantanusingh16/Perspective-Occupancy-Map},
version = {1.0.0},
year = {2022}
}
```
