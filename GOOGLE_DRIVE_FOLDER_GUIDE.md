# Kaggle_Computed_65Bands_DIAS — Google Drive Upload Guide

## How to upload
1. Upload this whole folder (`Kaggle_Computed_65Bands_DIAS`) to **My Drive**.
2. **Rename it to `Classification`** on Drive if running Colab notebooks that expect `/content/drive/My Drive/Classification/...`.
3. Open any notebook in Colab. Paths to `data/`, `*/models/` and `*/results/` resolve automatically.

## Clean Kaggle-Computed Structure (~7.9 GB)
Strictly models, clean notebooks, and results across all 12 modules — zero intermediate files:

```text
Kaggle_Computed_65Bands_DIAS/        (or rename to Classification on Drive)
├── README.md                        Project overview + Kaggle links
├── GOOGLE_DRIVE_FOLDER_GUIDE.md     This file
│
├── data/                            DIAS.mat, DIAS_ref.mat                      12 MB
│
├── baseline/                        AlexNet CNN / GFNet / ViT-UNet
│   ├── Model_training.ipynb
│   ├── Model_uncertainty_comparison.ipynb
│   ├── models/                      6 .keras (best + final)                     1.1 GB
│   └── results/                     training_plots/ scene_visualizations/ uncertainty_results/
├── credit/                          CREDIT evidential distillation
│   ├── Model_training_credit.ipynb
│   ├── models/                      3 .keras                                    571 MB
│   └── results/
├── ensemble/                        Deep ensembles + CreDE
│   ├── Model_training_ensembles.ipynb
│   ├── Model_uncertainty_CreDE.ipynb
│   ├── models/                      6 main + ensembles/ (15 members)            3.9 GB
│   └── results/
├── multicp/                         Multi-head conformal
│   ├── Model_training_multihead.ipynb
│   ├── Model_uncertainty_multicp.ipynb
│   ├── models/                      7 .keras                                    1.4 GB
│   └── results/
├── multicp_sacp/                    MultiCP + SACP
│   ├── Model_uncertainty_multicp_sacp.ipynb
│   └── results/window_3 … window_9/
├── sacp/                            Spatially aggregated CP (uses baseline weights)
│   ├── Model_sacp_comparison.ipynb
│   └── results/window_3 … window_9/
├── mambahsi/                        MambaHSI Small / Base / Large
│   ├── Model_training_mambahsi.ipynb
│   ├── Model_uncertainty_comparison_mambahsi.ipynb
│   ├── models/                      6 .keras                                    361 MB
│   └── results/
├── dofa/                            DOFA Spectral / Hiera ViT / Hiera Fusion
│   ├── Model_training.ipynb
│   ├── Model_uncertainty_comparison.ipynb
│   ├── models/                      6 .keras                                    168 MB
│   └── results/
├── georsclip/                       GeoRSCLIP text-head
│   ├── Model_training.ipynb
│   ├── Model_uncertainty_comparison.ipynb
│   ├── models/                      2 .keras                                    80 MB
│   └── results/
├── spatialgcn/                      Spatial GCN
│   ├── Model_training.ipynb
│   ├── Model_uncertainty_comparison.ipynb
│   ├── models/                      SpatialGCN_best/final.keras                 1.4 MB
│   └── results/
└── dapm/                            Diffusion-based DAPM (on baseline backbones)
    ├── Model_training_dapm_full.ipynb
    ├── Model_uncertainty_dapm_full.ipynb
    ├── Model_uncertainty_dapm_cp.ipynb
    ├── models/dapm_full_artifacts/  18 .weights.h5 (6 per backbone)            6.7 MB
    └── results/                     dapm_full_reports_all_models.xlsx, summary/per-class CSVs, plots
```

## Run order (dependencies)
- `baseline` → needed by `sacp`, `dapm`
- `ensemble` → needed by `credit` (distillation)
- `multicp/Model_training_multihead` → needed by `multicp/Model_uncertainty_multicp` and `multicp_sacp`
- All other folders are independent: train first, then run the uncertainty notebook.

## Key results (from the files in `results/`)
| Model | Test accuracy | Split CP coverage (95% target) |
| :--- | :---: | :---: |
| Baseline GFNet | 99.99% | 95.06% |
| DOFA Hiera ViT | 100.00% | 95.58% |
| GeoRSCLIP TextHead | 100.00% | 94.75% |
| MambaHSI Base | 99.99% | 95.96% |
| SpatialGCN | 97.05% | 95.24% |

DAPM certainty rate: AlexNet 99.83%, GFNet 99.90%, ViT-UNet 99.95% (see `dapm/results/dapm_full_summary_metrics.csv`).

## Also available online
- GitHub: https://github.com/DaneshSelwal/classification_dias
- Hugging Face weights: https://huggingface.co/DaneshSelwal/Classification-DIAS-65Bands
- Hugging Face data: https://huggingface.co/datasets/DaneshSelwal/Classification-DIAS-65Bands-Data

Author: Danesh Selwal (selwaldanesh@gmail.com) · Co-authored by Claude
