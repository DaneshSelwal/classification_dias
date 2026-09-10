# Classification Uncertainty Quantification Analysis (DIAS Dataset)

This repository contains baseline models and uncertainty quantification analysis trained on the DIAS (Dataset of Indian Agricultural Scenes) dataset.

## Repository Structure

- `baseline/`: Contains the Jupyter notebooks for baseline model training and uncertainty comparison.
- `baseline/models/`: Trained model weights (`.keras` format) for AlexNet CNN, GFNet, and ViT-UNet.
- `baseline/results/`: Output artifacts, including classification reports, confusion matrices, scene visualizations, training plots, and conformal prediction bounds.
- `data/`: The DIAS dataset and reference label files.
- `credit/`, `ensemble/`, `multicp/`, `sacp/`: Additional working directories for future comparative experiments.

## Models Evaluated

- **AlexNet CNN**
- **GFNet** (Global Filter Network)
- **ViT-UNet** (Vision Transformer UNet)

## Uncertainty Quantification

The uncertainty comparison evaluates models using multiple conformal prediction methods, capturing metrics such as Brier score, Expected Calibration Error (ECE), and per-class coverage guarantees.
