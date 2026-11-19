# Uncertainty methods in plain words

*Full report, chapter 3 of 6 (§7).* [Overview](../README.md) · [Report contents](README.md)

---

## 7. Uncertainty quantification: methods in plain words

### 7.1 Three families

| Family | Everyday analogy | Output per pixel | Guarantee? |
|---|---|---|---|
| **Conformal prediction**: Split, Class-conditional, Clustered, RC3P, RAPS, SACP, MultiCP, SCMCP | A forecaster who keeps score: *"on days I said this, I was right 95% of the time"* | A **set** of classes, e.g. {crop A} or {crop A, crop B} | **Yes**: the true class is in the set ≥ 95% of the time, provided calibration and test pixels are exchangeable |
| **Credal / ensemble**: CREDIT, CreDE | Ask five experts and report the range of their opinions | Lower/upper probability per class → aleatoric and epistemic uncertainty | No |
| **Generative + statistical test**: DAPM | Redraw the answer 30 times and test whether the top two classes are really different | p-value → certain / uncertain | No |

> **Two kinds of uncertainty.** *Aleatoric* uncertainty is ambiguity in the world: a mixed pixel really
> is half crop, half soil, and more data would not help. *Epistemic* uncertainty is a gap in the model's
> knowledge, and more or better data would reduce it. Credal methods try to separate the two; conformal
> methods report their combined effect as the size of a set.

### 7.2 Conformal prediction in three steps

<p align="center"><img src="figures/uncertainty/conformal_explainer.png" width="100%" alt="Conformal prediction explained"></p>

*__Box 1.__ How conformal prediction works, with illustrative (made-up) numbers.*

1. **Score.** For each calibration pixel, measure how "surprised" the model was by the true answer:
   $`s_i = 1 - \hat p_{y_i}(x_i)`$. A model that gave the true class 0.98 scores 0.02.
2. **Cut-off.** Find the value that 95% of calibration scores stay under, with a small finite-sample
   correction:

```math
\hat q = \text{Quantile}\left(s_1,\dots,s_n;\ \frac{\lceil (n+1)(1-\alpha)\rceil}{n}\right),\qquad \alpha = 0.05
```

3. **Set.** For a new pixel, keep every class the model would be no more surprised by:
   $`C(x) = \{\,k : \hat p_k(x) \ge 1 - \hat q\,\}`$.

New pixels are scored exactly like calibration pixels, so the true class falls under the cut-off
**95% of the time on average**. This holds whatever the model is. Every conformal folder in this
repository changes one ingredient: the **score** (RAPS, SAPS), the **grouping** used to compute the
cut-off (class-conditional, clustered, RC3P), the **neighbourhood** the score is computed over (SACP,
SCMCP), or the **number of heads** that must agree (MultiCP).

### 7.3 Method map

| Folder | Method | What it changes | Output | Read |
|---|---|---|---|---|
| [`baseline/`](../baseline/README.md) | Split CP, Class-conditional CP, Clustered CP, RC3P, RAPS | The score and the grouping | Prediction sets | ⭐ start here |
| [`sacp/`](../sacp/README.md) | Spatially-Aggregated CP | Averages each pixel's score with its neighbours' | Prediction sets | |
| [`multicp/`](../multicp/README.md) | Multi-head CP | 7 output heads; a class must pass all 7 | Prediction sets | |
| [`multicp_sacp/`](../multicp_sacp/README.md) | Spatial MultiCP (SCMCP) | Smooths probabilities in space, then MultiCP | Prediction sets | |
| [`ensemble/`](../ensemble/README.md) | Deep ensembles + CreDE | 5 independently trained models → credal intervals | AU / EU / TU, credal sets | |
| [`credit/`](../credit/README.md) | CREDIT | One student network learns the ensemble's intervals | AU / EU / TU, credal sets | |
| [`dapm/`](../dapm/README.md) | DAPM | VAE + domain alignment + diffusion sampling + Welch t-test | p-value per pixel | |
| [`dofa/`](../dofa/README.md), [`georsclip/`](../georsclip/README.md), [`mambahsi/`](../mambahsi/README.md), [`spatialgcn/`](../spatialgcn/README.md) | The five baseline CP methods | New backbones | Prediction sets | |

---

[← Models and classification results](02_models_and_classification.md) · [Report contents](README.md) · [Uncertainty results and discussion →](04_uncertainty_results.md)
