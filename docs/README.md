# Full report

[← Overview](../README.md)

The [main README](../README.md) gives the short version. This folder holds the full write-up, in six
chapters. Section numbers run continuously across chapters, and figures and tables keep one numbering,
so a reference such as "§8.3" or "Figure 16" means the same thing everywhere.

Each method folder also has its own README with the maths, a worked example and every result table.

### 1. [Data and experimental set-up](01_data_and_setup.md)

- [2. Why uncertainty matters for land-cover maps](01_data_and_setup.md#2-why-uncertainty-matters-for-land-cover-maps)
- [3. The data](01_data_and_setup.md#3-the-data)
  - [3.1 What the files contain](01_data_and_setup.md#31-what-the-files-contain)
  - [3.2 The scene and its classes](01_data_and_setup.md#32-the-scene-and-its-classes)
  - [3.3 Spectral behaviour: which classes look alike?](01_data_and_setup.md#33-spectral-behaviour-which-classes-look-alike)
  - [3.4 How much independent information do 65 bands carry?](01_data_and_setup.md#34-how-much-independent-information-do-65-bands-carry)
  - [3.5 A data-quality finding: stored values wrap round](01_data_and_setup.md#35-a-data-quality-finding-stored-values-wrap-round)
- [4. Experimental set-up](01_data_and_setup.md#4-experimental-set-up)
  - [4.1 Pipeline](01_data_and_setup.md#41-pipeline)
  - [4.2 Training recipe](01_data_and_setup.md#42-training-recipe)
  - [4.3 How pixels are split, and why it matters](01_data_and_setup.md#43-how-pixels-are-split-and-why-it-matters)
  - [4.4 Evaluation metrics (a glossary)](01_data_and_setup.md#44-evaluation-metrics-a-glossary)

### 2. [Models and classification results](02_models_and_classification.md)

- [5. Model architectures](02_models_and_classification.md#5-model-architectures)
  - [5.1 Baseline backbones](02_models_and_classification.md#51-baseline-backbones)
  - [5.2 Extended backbones (same five conformal methods)](02_models_and_classification.md#52-extended-backbones-same-five-conformal-methods)
- [6. Classification results](02_models_and_classification.md#6-classification-results)

### 3. [Uncertainty methods in plain words](03_uncertainty_methods.md)

- [7. Uncertainty quantification: methods in plain words](03_uncertainty_methods.md#7-uncertainty-quantification-methods-in-plain-words)
  - [7.1 Three families](03_uncertainty_methods.md#71-three-families)
  - [7.2 Conformal prediction in three steps](03_uncertainty_methods.md#72-conformal-prediction-in-three-steps)
  - [7.3 Method map](03_uncertainty_methods.md#73-method-map)

### 4. [Uncertainty results and discussion](04_uncertainty_results.md)

- [8. Uncertainty results](04_uncertainty_results.md#8-uncertainty-results)
  - [8.1 RQ1 · Does the 95% guarantee hold?](04_uncertainty_results.md#81-rq1--does-the-95-guarantee-hold)
  - [8.2 RQ2 · What do the prediction sets look like?](04_uncertainty_results.md#82-rq2--what-do-the-prediction-sets-look-like)
  - [8.3 RQ3 · Is coverage fair across classes?](04_uncertainty_results.md#83-rq3--is-coverage-fair-across-classes)
  - [8.4 RQ4 · Where on the map is the uncertainty?](04_uncertainty_results.md#84-rq4--where-on-the-map-is-the-uncertainty)
  - [8.5 RQ5 · How much of the map does each method call uncertain?](04_uncertainty_results.md#85-rq5--how-much-of-the-map-does-each-method-call-uncertain)
  - [8.6 RQ6 · Multi-head and spatial variants](04_uncertainty_results.md#86-rq6--multi-head-and-spatial-variants)
  - [8.7 RQ7 · What do ensembles say about the two kinds of uncertainty?](04_uncertainty_results.md#87-rq7--what-do-ensembles-say-about-the-two-kinds-of-uncertainty)
  - [8.8 Method scorecard](04_uncertainty_results.md#88-method-scorecard)
- [9. Discussion and recommendations](04_uncertainty_results.md#9-discussion-and-recommendations)

### 5. [Limitations and threats to validity](05_limitations.md)

- [10. Limitations and threats to validity](05_limitations.md#10-limitations-and-threats-to-validity)

### 6. [Reproducibility, citation and references](06_reproducibility_and_references.md)

- [11. Repository map and reproducibility](06_reproducibility_and_references.md#11-repository-map-and-reproducibility)
- [12. Citation, licence and references](06_reproducibility_and_references.md#12-citation-licence-and-references)
