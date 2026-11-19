# Limitations and threats to validity

*Full report, chapter 5 of 6 (§10).* [Overview](../README.md) · [Report contents](README.md)

---

## 10. Limitations and threats to validity

| # | Issue | Effect on results | Where |
|:-:|---|---|---|
| 1 | **Random pixel split**: every test patch contains training pixels (50.6 of 81 on average) | Accuracy, and every uncertainty metric computed on it, is optimistic. A spatially blocked split (whole parcels or tiles held out) is needed for a realistic estimate | [§4.3](01_data_and_setup.md#43-how-pixels-are-split-and-why-it-matters) |
| 2 | Stored values wrap round (signed vs unsigned 8-bit) | Bright pixels (mostly class 2) are fed to the models as dark ones | [§3.5](01_data_and_setup.md#35-a-data-quality-finding-stored-values-wrap-round) |
| 3 | AlexNet trained on split seed 10, while conformal calibration uses seed 42 | **75.2%** of AlexNet's calibration/evaluation pixels were in its training set, so its conformal numbers are not hold-out estimates | all CP folders |
| 4 | Near-perfect models + α = 0.05 | Sets are singletons or empty and do not single out errors | [§8.2](04_uncertainty_results.md#82-rq2--what-do-the-prediction-sets-look-like) |
| 5 | RAPS implemented without randomisation, and without the true class's own probability | Threshold collapses to 0 → always top-1 → coverage = accuracy | [baseline/](../baseline/README.md#raps) |
| 6 | MultiCP intersects independently calibrated heads | Joint coverage ≈ 0.95⁷ ≈ 70% | [multicp/](../multicp/README.md) |
| 7 | MultiCP / SCMCP maps treat label 7 as "unlabelled" (a hard-coded `y == 7` rule; the notebook's docstring calls label 7 "unlabelled") | Real class-7 parcels are greyed out in those maps | [multicp/](../multicp/README.md#33-maps) |
| 8 | CREDIT selects its best checkpoint on the test set | Small optimistic bias in CREDIT's test metrics | [credit/](../credit/README.md) |
| 9 | DOFA assumes 65 evenly spaced wavelengths, 0.40–2.50 µm | DOFA's wavelength conditioning is approximate | [dofa/](../dofa/README.md) |
| 10 | SACP / SCMCP smooth across calibration and test pixels | Exchangeability is bent, so the formal guarantee does not strictly apply | [sacp/](../sacp/README.md#4-caveats) |
| 11 | One scene, one sensor, one random split, no repeated seeds | No confidence intervals across runs; results may not transfer to other scenes | — |
| 12 | Class names and acquisition details unknown | Class interpretations in §3.2 are inferred | [§3.1](01_data_and_setup.md#31-what-the-files-contain) |

---

[← Uncertainty results and discussion](04_uncertainty_results.md) · [Report contents](README.md) · [Reproducibility, citation and references →](06_reproducibility_and_references.md)
