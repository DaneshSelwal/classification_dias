# Reproducibility, citation and references

*Full report, chapter 6 of 6 (§11–§12).* [Overview](../README.md) · [Report contents](README.md)

---

## 11. Repository map and reproducibility

```
Hyperspectral-LandCover-Uncertainty/
├── README.md                 ← overview: what we did and the end result
├── LICENSE                   ← Apache 2.0
├── docs/                     ← full report: six chapters + figures/
│   ├── 01_…06_*.md           ←   the six chapters (README.md = contents)
│   └── figures/              ←   figures made for this documentation
│       ├── dataset/          ←   data analysis (Figures 1–9)
│       ├── uncertainty/      ←   cross-method results (Figures 11–21)
│       └── maps/             ←   uncertainty maps extracted from the result workbooks
├── baseline/   credit/   ensemble/   dapm/
├── multicp/    sacp/     multicp_sacp/
├── dofa/       georsclip/  mambahsi/  spatialgcn/
│   ├── README.md             ← method write-up: intuition, maths, results, caveats
│   ├── *.ipynb               ← executed notebooks (Kaggle GPU)
│   └── results/              ← CSV / JSON / PNG / XLSX outputs (the source of every number)
└── scripts/                  ← verification helpers
```

| Resource | Link |
|---|---|
| Data (`DIAS.mat`, `DIAS_ref.mat`) | 🤗 [datasets/DaneshSelwal/Classification-DIAS-65Bands-Data](https://huggingface.co/datasets/DaneshSelwal/Classification-DIAS-65Bands-Data) |
| Trained weights | 🤗 [DaneshSelwal/Classification-DIAS-65Bands](https://huggingface.co/DaneshSelwal/Classification-DIAS-65Bands) |
| Browsable workspace | 🤗 [spaces/DaneshSelwal/Classification-DIAS-65Bands-Workspace](https://huggingface.co/spaces/DaneshSelwal/Classification-DIAS-65Bands-Workspace) |
| Kaggle data | [kaggle.com/datasets/daneshselwal/dias-dataset-k01](https://www.kaggle.com/datasets/daneshselwal/dias-dataset-k01) |
| Umbrella project (11 datasets) | [Classification_Uncertainty_Quantification_Analysis](https://github.com/DaneshSelwal/Classification_Uncertainty_Quantification_Analysis) |

**To reproduce:** run each folder's training notebook, then its uncertainty notebook, on a GPU (Kaggle
T4/P100 was used). Seeds are fixed (§4.2). Paths assume Kaggle input datasets; for a local run, edit the
configuration cell (Section 2 of each notebook). The documentation figures were generated from the
committed `results/` files and the Hugging Face data.

---

## 12. Citation, licence and references

**Licence.** Released under the [Apache License 2.0](../LICENSE).

**Citation.** If you use this work, please cite:

```bibtex
@misc{selwal2026dias_uq,
  author       = {Selwal, Danesh},
  title        = {Knowing When the Map Is Wrong: Uncertainty Quantification for
                  65-Band DIAS Hyperspectral Land-Cover Classification},
  year         = {2026},
  howpublished = {\url{https://github.com/DaneshSelwal/Hyperspectral-LandCover-Uncertainty}}
}
```

**References.**

1. Angelopoulos, A. N., & Bates, S. (2023). *Conformal Prediction: A Gentle Introduction.* Foundations and Trends in ML.
2. Vovk, V., Gammerman, A., & Shafer, G. (2005). *Algorithmic Learning in a Random World.* Springer.
3. Angelopoulos, A. N., Bates, S., Malik, J., & Jordan, M. I. (2021). *Uncertainty Sets for Image Classifiers using Conformal Prediction* (RAPS). ICLR.
4. Romano, Y., Sesia, M., & Candès, E. (2020). *Classification with Valid and Adaptive Coverage* (APS). NeurIPS.
5. Ding, T., et al. (2023). *Class-Conditional Conformal Prediction with Many Classes* (Clustered CP). NeurIPS.
6. Shi, Y., et al. (2024). *Conformal Prediction for Class-wise Coverage via Augmented Label Rank Calibration* (RC3P). NeurIPS.
7. Huang, J., et al. (2024). *Conformal Prediction for Deep Classifier via Label Ranking* (SAPS). ICML.
8. Lakshminarayanan, B., Pritzel, A., & Blundell, C. (2017). *Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles.* NeurIPS.
9. Hüllermeier, E., & Waegeman, W. (2021). *Aleatoric and Epistemic Uncertainty in Machine Learning.* Machine Learning 110.
10. Ganin, Y., et al. (2016). *Domain-Adversarial Training of Neural Networks.* JMLR.
11. Ho, J., Jain, A., & Abbeel, P. (2020). *Denoising Diffusion Probabilistic Models.* NeurIPS.
12. Rao, Y., et al. (2021). *Global Filter Networks for Image Classification* (GFNet). NeurIPS.
13. Dosovitskiy, A., et al. (2021). *An Image is Worth 16×16 Words* (ViT). ICLR.
14. Xiong, Z., et al. (2024). *Neural Plasticity-Inspired Foundation Model for Observing the Earth Crossing Modalities* (DOFA).
15. Gu, A., & Dao, T. (2023). *Mamba: Linear-Time Sequence Modeling with Selective State Spaces.*
16. Kipf, T. N., & Welling, M. (2017). *Semi-Supervised Classification with Graph Convolutional Networks.* ICLR.
17. Roberts, D. R., et al. (2017). *Cross-validation strategies for data with temporal, spatial, hierarchical, or phylogenetic structure.* Ecography 40.
18. Multi-CP reference implementation: [github.com/yamtawa/Multi-CP](https://github.com/yamtawa/Multi-CP).

---

[← Limitations and threats to validity](05_limitations.md) · [Report contents](README.md) · [Back to overview →](../README.md)
