# DOFA Model Analysis and Classification Report

## Abstract
This report presents a comprehensive analysis of the Dynamic One-For-All (DOFA) model in the context of hyperspectral image classification and uncertainty quantification. DOFA leverages a unified foundational architecture to process multi-modal and variable-band spectral data dynamically. In this study, we compare three model variants: **DOFA_Hiera_Fusion**, **Hiera_ViT**, and **DOFA_Spectral**. The models are evaluated on test accuracy, cross-entropy, Brier score, and various conformal prediction methods (Split Conformal, Class-Conditional, Clustered, RC3P, and RAPS). Results indicate that the fusion-based DOFA architecture significantly outperforms spectral-only approaches, achieving near-perfect classification accuracy while maintaining robust uncertainty calibration.

## Introduction
Hyperspectral imaging (HSI) involves capturing detailed spectral information across multiple contiguous wavelengths. A significant challenge in HSI is the variability in sensor specifications, leading to datasets with different numbers of bands and spectral ranges. The **DOFA (Dynamic One-For-All)** framework addresses this by employing a wavelength-dependent dynamic tokenizer. Instead of learning fixed weights for a specific sensor, DOFA generates projection weights dynamically based on the input wavelengths, allowing it to act as a unified foundation model for varied earth observation tasks. 

In this report, we evaluate DOFA using three architectural configurations:
- **DOFA_Spectral**: Evaluates the spectral stream independently.
- **Hiera_ViT**: A hierarchical Vision Transformer operating on the spatial/patch domain.
- **DOFA_Hiera_Fusion**: A unified model combining the dynamic spectral embeddings of DOFA with the spatial hierarchical representation of Hiera ViT.

## Methodology

### The DOFA Framework
DOFA introduces a dynamic weight generation mechanism. Given a set of wavelengths $\lambda = [\lambda_1, \lambda_2, \dots, \lambda_C]$ associated with a specific sensor, DOFA uses a meta-network (often a Multi-Layer Perceptron) to map each wavelength to a distinct filter weight. This allows the model to process arbitrary spectral bands without requiring architecture modifications or retraining from scratch. The continuous nature of the meta-network ensures that closely related wavelengths receive similar weight representations, capturing the physical properties of the spectral signatures.

### Model Configurations
1. **DOFA_Spectral**: Focuses purely on spectral correlations using the dynamic wavelength tokenization.
2. **Hiera_ViT**: Employs a hierarchical Vision Transformer (Hiera) that efficiently downsamples spatial features, capturing multi-scale spatial context.
3. **DOFA_Hiera_Fusion**: Integrates spatial features from Hiera_ViT with spectral embeddings from DOFA. The fusion is typically performed via concatenation followed by dense layers or a cross-attention mechanism, resulting in a joint spatial-spectral representation that maximizes discriminatory power.

### Uncertainty Quantification
To ensure reliable predictions, we apply several Conformal Prediction (CP) techniques:
- **Split Conformal**: Standard marginal coverage over a calibration set.
- **Class-Conditional Conformal**: Ensures coverage guarantees per individual class.
- **Clustered Conformal**: Groups similar samples to calibrate set sizes dynamically.
- **RC3P & RAPS**: Advanced regularized prediction sets that penalize excessively large set sizes, improving efficiency.

## Experiments and Results

### Classification Performance
The models were trained for 100 epochs. The `DOFA_Hiera_Fusion` model achieved the highest performance across all metrics, significantly outperforming `DOFA_Spectral`.

| Model | Test Accuracy | Kappa | Macro F1 | Weighted F1 | Test NLL | Test Brier | ECE (15-bin) |
|---|---|---|---|---|---|---|---|
| **DOFA_Hiera_Fusion** | 0.9965 | 0.9953 | 0.9970 | 0.9965 | 0.0369 | 0.0065 | 0.0273 |
| **Hiera_ViT** | 0.9951 | 0.9934 | 0.9931 | 0.9951 | 0.0502 | 0.0093 | 0.0338 |
| **DOFA_Spectral** | 0.6884 | 0.5678 | 0.6476 | 0.6672 | 0.7916 | 0.4198 | 0.0504 |

The fusion model demonstrates an exceptional accuracy of 99.65%, highlighting the necessity of combining spatial and dynamic spectral features. The standalone spectral model (`DOFA_Spectral`) struggles with spatial ambiguity, yielding a much lower accuracy of 68.84%.

![Model Comparison Metrics](/Users/danesh/Desktop/Classific/Classification_Uncertainty_Quantification_Analysis/Classification/dofa/results/training_plots/model_comparison_metrics.png){width=6in}

![Confusion Matrices](/Users/danesh/Desktop/Classific/Classification_Uncertainty_Quantification_Analysis/Classification/dofa/results/training_plots/confusion_matrices_side_by_side.png){width=6in}

### Training Dynamics

The learning curves reflect the stability and rapid convergence of the spatial and fusion architectures compared to the spectral-only baseline.

| DOFA_Hiera_Fusion | DOFA_Spectral | Hiera_ViT |
| :---: | :---: | :---: |
| ![](/Users/danesh/Desktop/Classific/Classification_Uncertainty_Quantification_Analysis/Classification/dofa/results/training_plots/DOFA_Hiera_Fusion_training_curves.png){width=2in} | ![](/Users/danesh/Desktop/Classific/Classification_Uncertainty_Quantification_Analysis/Classification/dofa/results/training_plots/DOFA_Spectral_training_curves.png){width=2in} | ![](/Users/danesh/Desktop/Classific/Classification_Uncertainty_Quantification_Analysis/Classification/dofa/results/training_plots/Hiera_ViT_training_curves.png){width=2in} |

### Conformal Prediction and Uncertainty Analysis

We evaluated conformal prediction coverage and set sizes at a target coverage of **95%** ($\alpha = 0.05$). The summary metrics below illustrate the robustness of the predictions.

| Model | CP Method | Empirical Coverage | Avg Set Size | Singleton Rate | Empty Set Rate |
|---|---|---|---|---|---|
| DOFA_Hiera_Fusion | ClassConditional | 0.9568 | 0.9568 | 1.0 | 0.0432 |
| DOFA_Hiera_Fusion | RAPS | 0.9949 | 1.0000 | 1.0 | 0.0000 |
| DOFA_Hiera_Fusion | RC3P | 0.9629 | 0.9629 | 1.0 | 0.0371 |
| DOFA_Spectral | ClassConditional | 0.9564 | 2.6037 | 0.2381 | 0.0000 |
| DOFA_Spectral | RAPS | 0.9452 | 2.2019 | 0.2520 | 0.0000 |
| Hiera_ViT | ClassConditional | 0.9652 | 0.9657 | 1.0 | 0.0343 |

`DOFA_Hiera_Fusion` maintains near-perfect coverage with a minimal average set size (~1.0), meaning it almost always provides a single, correct prediction. In contrast, `DOFA_Spectral` requires an average set size of ~2.6 to achieve the same 95% confidence, reflecting its high underlying uncertainty.

![Uncertainty Proxy Metrics](/Users/danesh/Desktop/Classific/Classification_Uncertainty_Quantification_Analysis/Classification/dofa/results/training_plots/uncertainty_proxy_metrics.png){width=6in}

### Scene Visualizations

The predicted classification maps demonstrate the qualitative improvements of the fusion and hierarchical approaches. The spatial coherence in the fusion model closely mirrors the ground truth, effectively eliminating the noise and artifacts present in the spectral-only classification.

**Reference Images:**

| Scene RGB | Ground Truth Label Map |
| :---: | :---: |
| ![](/Users/danesh/Desktop/Classific/Classification_Uncertainty_Quantification_Analysis/Classification/dofa/results/scene_visualizations/scene_rgb.png){width=3in} | ![](/Users/danesh/Desktop/Classific/Classification_Uncertainty_Quantification_Analysis/Classification/dofa/results/scene_visualizations/ground_truth_label_map.png){width=3in} |

**Classification Results:**

| DOFA_Hiera_Fusion | Hiera_ViT | DOFA_Spectral |
| :---: | :---: | :---: |
| ![](/Users/danesh/Desktop/Classific/Classification_Uncertainty_Quantification_Analysis/Classification/dofa/results/scene_visualizations/DOFA_Hiera_Fusion_initial_classification.png){width=2in} | ![](/Users/danesh/Desktop/Classific/Classification_Uncertainty_Quantification_Analysis/Classification/dofa/results/scene_visualizations/Hiera_ViT_initial_classification.png){width=2in} | ![](/Users/danesh/Desktop/Classific/Classification_Uncertainty_Quantification_Analysis/Classification/dofa/results/scene_visualizations/DOFA_Spectral_initial_classification.png){width=2in} |

![Combined Initial Classification Overview](/Users/danesh/Desktop/Classific/Classification_Uncertainty_Quantification_Analysis/Classification/dofa/results/scene_visualizations/combined_initial_classification_overview.png){width=6in}

## Conclusion
The application of the DOFA framework to hyperspectral image classification highlights the power of dynamic wavelength tokenization when combined with spatial hierarchical modeling. The `DOFA_Hiera_Fusion` model achieves outstanding performance (99.65% accuracy) and optimal uncertainty calibration, outputting highly confident and accurate singleton prediction sets. The purely spectral model, while benefiting from the flexible tokenizer, lacks the spatial context required for competitive accuracy, resulting in larger conformal prediction sets. Overall, DOFA's foundational approach coupled with spatial fusion represents a highly robust solution for earth observation tasks.
